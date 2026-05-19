package ds4

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"github.com/NimbleMarkets/ds4go"
	"github.com/NimbleMarkets/ds4go/ds4api"
	"github.com/NimbleMarkets/ds4go/dsml"
)

// languageModel implements fantasy.LanguageModel for the DS4 provider.
// Native in-process inference — no HTTP round-trips, zero-copy token
// handling, persistent KV-cache session reuse for agent loops.
//
// Prompt rendering uses libds4's own chat helpers for the chat envelope
// and the ds4go dsml package for DSML tool-calling markup; output is
// parsed with dsml.ParseCompletion (Generate) and dsml.StreamDecoder
// (Stream).
type languageModel struct {
	provider         string
	modelID          string
	getEngine        func() (*ds4api.Engine, error)
	defaultThinkMode ds4api.ThinkMode
	objectMode       fantasy.ObjectMode
}

func newLanguageModel(provider, modelID string, getEngine func() (*ds4api.Engine, error), thinkMode ds4api.ThinkMode, om fantasy.ObjectMode) *languageModel {
	return &languageModel{
		provider:         provider,
		modelID:          modelID,
		getEngine:        getEngine,
		defaultThinkMode: thinkMode,
		objectMode:       om,
	}
}

func (m *languageModel) Model() string    { return m.modelID }
func (m *languageModel) Provider() string { return m.provider }

// toolCallID derives a stable per-completion tool-call ID from a tool
// name and its 0-based position in the completion.
func toolCallID(name string, index int) string {
	return fmt.Sprintf("%s-%d", name, index)
}

// ---------------------------------------------------------------------------
// Prompt building
// ---------------------------------------------------------------------------

// buildPrompt renders the fantasy Call into ds4 prompt tokens. It
// delegates the chat envelope, DSML tool markup, and ThinkMax prefix
// handling to ds4.BuildChatPrompt — the same builder ds4go's own tooling
// uses. The caller owns the returned tokens and must Free them.
func buildPrompt(engine *ds4api.Engine, call fantasy.Call, defaultThink ds4api.ThinkMode) (*ds4api.Tokens, ds4api.ThinkMode, error) {
	thinkMode := defaultThink
	if po := providerOptions(call); po != nil && po.ThinkMode != nil {
		thinkMode = *po.ThinkMode
	}

	tools, err := dsmlTools(call.Tools)
	if err != nil {
		return nil, thinkMode, err
	}

	tokens, err := ds4.BuildChatPrompt(
		engine,
		collectSystemMessages(call.Prompt),
		tools,
		chatHistory(call.Prompt),
		thinkMode,
	)
	if err != nil {
		return nil, thinkMode, fmt.Errorf("ds4: build prompt: %w", err)
	}
	return tokens, thinkMode, nil
}

func collectSystemMessages(prompt fantasy.Prompt) string {
	var parts []string
	for _, msg := range prompt {
		if msg.Role != fantasy.MessageRoleSystem {
			continue
		}
		for _, part := range msg.Content {
			if text, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
				parts = append(parts, text.Text)
			}
		}
	}
	return strings.Join(parts, "\n\n")
}

// dsmlTools converts fantasy function tools into dsml.Tool definitions.
// Non-function tools are skipped.
func dsmlTools(tools []fantasy.Tool) ([]dsml.Tool, error) {
	out := make([]dsml.Tool, 0, len(tools))
	for _, tool := range tools {
		ft, ok := tool.(fantasy.FunctionTool)
		if !ok {
			continue
		}
		var params json.RawMessage
		if ft.InputSchema != nil {
			b, err := json.Marshal(ft.InputSchema)
			if err != nil {
				return nil, fmt.Errorf("ds4: marshal tool %q schema: %w", ft.Name, err)
			}
			params = b
		}
		out = append(out, dsml.Tool{
			Name:        ft.Name,
			Description: ft.Description,
			Parameters:  params,
		})
	}
	return out, nil
}

// chatHistory converts the non-system fantasy messages into ds4
// ChatMessages. System messages are handled separately via
// collectSystemMessages. Each tool-result part becomes its own tool
// message; ds4.BuildChatPrompt collapses consecutive tool messages into
// a single user turn, preserving invoke order.
func chatHistory(prompt fantasy.Prompt) []ds4.ChatMessage {
	var msgs []ds4.ChatMessage
	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			continue

		case fantasy.MessageRoleUser:
			msgs = append(msgs, ds4.ChatMessage{Role: "user", Content: messageText(msg)})

		case fantasy.MessageRoleAssistant:
			cm := ds4.ChatMessage{Role: "assistant"}
			for _, part := range msg.Content {
				switch part.GetType() {
				case fantasy.ContentTypeText:
					if t, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
						cm.Content += t.Text
					}
				case fantasy.ContentTypeToolCall:
					if tc, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part); ok {
						cm.ToolCalls = append(cm.ToolCalls, ds4.ToolCall{
							ID:        tc.ToolCallID,
							Name:      tc.ToolName,
							Arguments: tc.Input,
						})
					}
				}
			}
			msgs = append(msgs, cm)

		case fantasy.MessageRoleTool:
			for _, part := range msg.Content {
				if r, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part); ok {
					msgs = append(msgs, ds4.ChatMessage{
						Role:       "tool",
						Content:    toolResultText(r),
						ToolCallID: r.ToolCallID,
					})
				}
			}
		}
	}
	return msgs
}

// messageText concatenates the text parts of a message.
func messageText(msg fantasy.Message) string {
	var sb strings.Builder
	for _, part := range msg.Content {
		if text, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
			sb.WriteString(text.Text)
		}
	}
	return sb.String()
}

func toolResultText(result fantasy.ToolResultPart) string {
	switch out := result.Output.(type) {
	case fantasy.ToolResultOutputContentText:
		return out.Text
	case fantasy.ToolResultOutputContentError:
		return out.Error.Error()
	default:
		return ""
	}
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

// Generate implements fantasy.LanguageModel.
func (m *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	engine, err := m.getEngine()
	if err != nil {
		return nil, fmt.Errorf("ds4: engine: %w", err)
	}

	tokens, thinkMode, err := buildPrompt(engine, call, m.defaultThinkMode)
	if err != nil {
		return nil, err
	}
	defer tokens.Free()

	session, err := engine.NewSession(sessionCtxSize(call))
	if err != nil {
		return nil, fmt.Errorf("ds4: create session: %w", err)
	}
	defer session.Close()

	genOpts := buildGenerateOptions(call)
	genCtx, cancel := generationContext(ctx)
	defer cancel()
	genOpts.Context = genCtx

	var output strings.Builder
	var generated int
	genOpts.OnToken = func(token int) {
		select {
		case <-genCtx.Done():
			return
		default:
		}
		generated++
		if text, decErr := engine.TokenText(token); decErr == nil {
			output.WriteString(text)
		}
	}

	gen := ds4.Generator{Engine: engine, Session: session}
	_, genErr := gen.GenerateTokens(tokens, genOpts)
	if genErr != nil && !errors.Is(genErr, ds4.ErrContextFull) &&
		!errors.Is(genErr, context.Canceled) && !errors.Is(genErr, context.DeadlineExceeded) {
		return nil, fmt.Errorf("ds4: generate: %w", genErr)
	}

	parsed, err := dsml.ParseCompletion(output.String(), thinkMode != ds4api.ThinkNone)
	if err != nil {
		return nil, fmt.Errorf("ds4: parse completion: %w", err)
	}

	content := make([]fantasy.Content, 0, 2+len(parsed.ToolCalls))
	if parsed.ReasoningContent != "" {
		content = append(content, fantasy.ReasoningContent{Text: parsed.ReasoningContent})
	}
	if parsed.Content != "" {
		content = append(content, fantasy.TextContent{Text: parsed.Content})
	}
	for i, tc := range parsed.ToolCalls {
		content = append(content, fantasy.ToolCallContent{
			ToolCallID: toolCallID(tc.Name, i),
			ToolName:   tc.Name,
			Input:      tc.Arguments,
		})
	}

	finishReason := fantasy.FinishReasonStop
	if len(parsed.ToolCalls) > 0 {
		finishReason = fantasy.FinishReasonToolCalls
	} else if (genOpts.MaxTokens > 0 && generated >= genOpts.MaxTokens) || errors.Is(genErr, ds4.ErrContextFull) {
		finishReason = fantasy.FinishReasonLength
	}

	inputTokens := tokens.Len()
	return &fantasy.Response{
		Content:      content,
		FinishReason: finishReason,
		Usage: fantasy.Usage{
			InputTokens:  int64(inputTokens),
			OutputTokens: int64(generated),
			TotalTokens:  int64(inputTokens + generated),
		},
	}, nil
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

// Stream implements fantasy.LanguageModel.
func (m *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	engine, err := m.getEngine()
	if err != nil {
		return nil, fmt.Errorf("ds4: engine: %w", err)
	}

	tokens, thinkMode, err := buildPrompt(engine, call, m.defaultThinkMode)
	if err != nil {
		return nil, err
	}

	session, err := engine.NewSession(sessionCtxSize(call))
	if err != nil {
		tokens.Free()
		return nil, fmt.Errorf("ds4: create session: %w", err)
	}

	genOpts := buildGenerateOptions(call)
	genCtx, cancel := generationContext(ctx)
	genOpts.Context = genCtx

	return func(yield func(fantasy.StreamPart) bool) {
		defer cancel()
		defer session.Close()
		defer tokens.Free()

		decoder := dsml.NewStreamDecoder(thinkMode != ds4api.ThinkNone)
		emitter := newStreamEmitter(yield)

		var (
			generated int
			stopped   bool
		)

		genOpts.OnToken = func(token int) {
			if stopped {
				return
			}
			select {
			case <-genCtx.Done():
				return
			default:
			}
			generated++
			text, decErr := engine.TokenText(token)
			if decErr != nil {
				return
			}
			for _, ev := range decoder.Write(text) {
				if !emitter.event(ev) {
					stopped = true
					return
				}
			}
		}

		gen := ds4.Generator{Engine: engine, Session: session}
		_, genErr := gen.GenerateTokens(tokens, genOpts)
		if genErr != nil && !errors.Is(genErr, ds4.ErrContextFull) &&
			!errors.Is(genErr, context.Canceled) && !errors.Is(genErr, context.DeadlineExceeded) {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: fmt.Errorf("ds4: generate: %w", genErr),
			})
			return
		}
		if stopped {
			return
		}

		finalEvents, parsed, closeErr := decoder.Close()
		if closeErr != nil {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: fmt.Errorf("ds4: parse completion: %w", closeErr),
			})
			return
		}
		for _, ev := range finalEvents {
			if !emitter.event(ev) {
				return
			}
		}
		if !emitter.finish() {
			return
		}

		finishReason := fantasy.FinishReasonStop
		if len(parsed.ToolCalls) > 0 {
			finishReason = fantasy.FinishReasonToolCalls
		} else if (genOpts.MaxTokens > 0 && generated >= genOpts.MaxTokens) || errors.Is(genErr, ds4.ErrContextFull) {
			finishReason = fantasy.FinishReasonLength
		}

		inputTokens := tokens.Len()
		yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeFinish,
			Usage: fantasy.Usage{
				InputTokens:  int64(inputTokens),
				OutputTokens: int64(generated),
				TotalTokens:  int64(inputTokens + generated),
			},
			FinishReason: finishReason,
		})
	}, nil
}

// ---------------------------------------------------------------------------
// Stream event translation
// ---------------------------------------------------------------------------

// streamEmitter translates dsml.StreamEvents into fantasy StreamParts,
// tracking which reasoning / text / tool blocks are open so the correct
// start and end parts bracket every delta. The final ToolCall part takes
// its input from the authoritative dsml.EventToolCallEnd.Arguments.
type streamEmitter struct {
	yield func(fantasy.StreamPart) bool

	reasoningOpen bool
	reasoningID   string
	reasoningSeq  int

	textOpen bool
	textID   string
	textSeq  int

	toolName map[int]string
}

func newStreamEmitter(yield func(fantasy.StreamPart) bool) *streamEmitter {
	return &streamEmitter{
		yield:    yield,
		toolName: map[int]string{},
	}
}

func (e *streamEmitter) event(ev dsml.StreamEvent) bool {
	switch ev.Type {
	case dsml.EventReasoningDelta:
		if !e.reasoningOpen {
			e.reasoningID = fmt.Sprintf("reasoning-%d", e.reasoningSeq)
			e.reasoningSeq++
			if !e.yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningStart, ID: e.reasoningID}) {
				return false
			}
			e.reasoningOpen = true
		}
		return e.yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningDelta, ID: e.reasoningID, Delta: ev.Delta})

	case dsml.EventContentDelta:
		if !e.closeReasoning() {
			return false
		}
		if !e.textOpen {
			e.textID = fmt.Sprintf("text-%d", e.textSeq)
			e.textSeq++
			if !e.yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart, ID: e.textID}) {
				return false
			}
			e.textOpen = true
		}
		return e.yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextDelta, ID: e.textID, Delta: ev.Delta})

	case dsml.EventToolCallStart:
		if !e.closeReasoning() || !e.closeText() {
			return false
		}
		e.toolName[ev.Index] = ev.Name
		return e.yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeToolInputStart,
			ID:           toolCallID(ev.Name, ev.Index),
			ToolCallName: ev.Name,
		})

	case dsml.EventToolCallArgumentsDelta:
		return e.yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeToolInputDelta,
			ID:    toolCallID(e.toolName[ev.Index], ev.Index),
			Delta: ev.Delta,
		})

	case dsml.EventToolCallEnd:
		id := toolCallID(e.toolName[ev.Index], ev.Index)
		if !e.yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeToolInputEnd, ID: id}) {
			return false
		}
		return e.yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolCall,
			ID:            id,
			ToolCallName:  e.toolName[ev.Index],
			ToolCallInput: ev.Arguments,
		})
	}
	return true
}

// finish closes any reasoning or text block still open at end of stream.
func (e *streamEmitter) finish() bool {
	return e.closeReasoning() && e.closeText()
}

func (e *streamEmitter) closeReasoning() bool {
	if !e.reasoningOpen {
		return true
	}
	e.reasoningOpen = false
	return e.yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningEnd, ID: e.reasoningID})
}

func (e *streamEmitter) closeText() bool {
	if !e.textOpen {
		return true
	}
	e.textOpen = false
	return e.yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd, ID: e.textID})
}

// ---------------------------------------------------------------------------
// GenerateObject / StreamObject
// ---------------------------------------------------------------------------

// GenerateObject implements fantasy.LanguageModel.
func (m *languageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	switch m.objectMode {
	case fantasy.ObjectModeText:
		return object.GenerateWithText(ctx, m, call)
	default:
		return object.GenerateWithTool(ctx, m, call)
	}
}

// StreamObject implements fantasy.LanguageModel.
func (m *languageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	switch m.objectMode {
	case fantasy.ObjectModeText:
		return object.StreamWithText(ctx, m, call)
	default:
		return object.StreamWithTool(ctx, m, call)
	}
}

// ---------------------------------------------------------------------------
// Generation options
// ---------------------------------------------------------------------------

func providerOptions(call fantasy.Call) *ProviderOptions {
	if v, ok := call.ProviderOptions[Name]; ok {
		if po, ok := v.(*ProviderOptions); ok {
			return po
		}
	}
	return nil
}

func sessionCtxSize(call fantasy.Call) int {
	po := providerOptions(call)
	if po != nil && po.CtxSize != nil {
		return *po.CtxSize
	}
	return 32768
}

func buildGenerateOptions(call fantasy.Call) ds4.GenerateOptions {
	po := providerOptions(call)
	opts := ds4.GenerateOptions{
		StopOnEOS: true,
	}

	maxTokens := 4096
	if po != nil && po.MaxTokens != nil {
		maxTokens = int(*po.MaxTokens)
	} else if call.MaxOutputTokens != nil {
		maxTokens = int(*call.MaxOutputTokens)
	}
	opts.MaxTokens = maxTokens

	temperature := float32(ds4api.DefaultTemperature)
	if po != nil && po.Temperature != nil {
		temperature = *po.Temperature
	} else if call.Temperature != nil {
		temperature = float32(*call.Temperature)
	}
	opts.Temperature = temperature

	topP := ds4api.DefaultTopP
	if po != nil && po.TopP != nil {
		topP = *po.TopP
	} else if call.TopP != nil {
		topP = float32(*call.TopP)
	}
	opts.TopP = topP

	minP := ds4api.DefaultMinP
	if po != nil && po.MinP != nil {
		minP = *po.MinP
	}
	opts.MinP = minP

	topK := 0
	if po != nil && po.TopK != nil {
		topK = int(*po.TopK)
	} else if call.TopK != nil {
		topK = int(*call.TopK)
	}
	opts.TopK = topK

	if po != nil && po.Seed != nil {
		opts.Seed = *po.Seed
	}

	return opts
}

func generationContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithCancel(ctx)
}
