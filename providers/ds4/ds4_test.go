package ds4

import (
	"context"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/NimbleMarkets/ds4go/ds4api"
	"github.com/NimbleMarkets/ds4go/dsml"
)

// TestProviderNew tests basic provider creation.
func TestProviderNew(t *testing.T) {
	p, err := New(
		WithName("test-ds4"),
	)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if p.Name() != "test-ds4" {
		t.Errorf("Name() = %q, want %q", p.Name(), "test-ds4")
	}
}

// TestProviderName verifies the default provider name.
func TestProviderName(t *testing.T) {
	p, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if p.Name() != Name {
		t.Errorf("Name() = %q, want %q", p.Name(), Name)
	}
}

// TestProviderOptions verifies provider option type registration.
func TestProviderOptions(t *testing.T) {
	po := &ProviderOptions{}
	po.Options() // marker method

	opts := fantasy.ProviderOptions{
		Name: po,
	}
	if v, ok := opts[Name]; !ok {
		t.Fatal("provider options not registered under provider name")
	} else if _, ok := v.(*ProviderOptions); !ok {
		t.Fatalf("provider options has wrong type: %T", v)
	}
}

// TestLanguageModelInterface verifies the language model satisfies the
// fantasy.LanguageModel interface.
func TestLanguageModelInterface(t *testing.T) {
	lm := newLanguageModel(
		"ds4",
		"deepseek-v4-flash",
		// engine getter is unused here; interface satisfaction only.
		func() (*ds4api.Engine, error) { return nil, nil },
		ds4api.ThinkNone,
		fantasy.ObjectModeTool,
	)

	if lm.Model() != "deepseek-v4-flash" {
		t.Errorf("Model() = %q, want %q", lm.Model(), "deepseek-v4-flash")
	}
	if lm.Provider() != "ds4" {
		t.Errorf("Provider() = %q, want %q", lm.Provider(), "ds4")
	}

	// Verify it satisfies the interface at compile time.
	var _ fantasy.LanguageModel = lm
}

// TestLanguageModelContext verifies LanguageModel returns a model without
// building the engine (engine creation is deferred to Generate/Stream).
func TestLanguageModelContext(t *testing.T) {
	p, err := New(WithName("test-ds4"))
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	lm, err := p.LanguageModel(context.Background(), "deepseek-v4-flash")
	if err != nil {
		t.Fatalf("LanguageModel() error: %v", err)
	}
	if lm.Model() != "deepseek-v4-flash" {
		t.Errorf("Model() = %q, want %q", lm.Model(), "deepseek-v4-flash")
	}
}

// TestLanguageModelDefaultModelID verifies an empty model ID falls back
// to the default model.
func TestLanguageModelDefaultModelID(t *testing.T) {
	p, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	lm, err := p.LanguageModel(context.Background(), "")
	if err != nil {
		t.Fatalf("LanguageModel() error: %v", err)
	}
	if lm.Model() != defaultModelID {
		t.Errorf("Model() = %q, want %q", lm.Model(), defaultModelID)
	}
}

// TestBuildGenerateOptions verifies option construction from call fields.
func TestBuildGenerateOptions(t *testing.T) {
	call := fantasy.Call{
		MaxOutputTokens: new(int64(2048)),
		Temperature:     new(0.7),
		TopP:            new(0.9),
	}

	opts := buildGenerateOptions(call)
	if opts.MaxTokens != 2048 {
		t.Errorf("MaxTokens = %d, want 2048", opts.MaxTokens)
	}
	if opts.Temperature != 0.7 {
		t.Errorf("Temperature = %f, want 0.7", opts.Temperature)
	}
	if opts.TopP != 0.9 {
		t.Errorf("TopP = %f, want 0.9", opts.TopP)
	}
	if !opts.StopOnEOS {
		t.Error("StopOnEOS should be true")
	}
}

// TestBuildGenerateOptionsWithProviderOptions verifies provider options
// override call-level fields.
func TestBuildGenerateOptionsWithProviderOptions(t *testing.T) {
	temp := float32(0.3)
	maxTok := int32(100)
	call := fantasy.Call{
		MaxOutputTokens: new(int64(2048)),
		Temperature:     new(0.7),
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ProviderOptions{
				Temperature: &temp,
				MaxTokens:   &maxTok,
			},
		},
	}

	opts := buildGenerateOptions(call)
	if opts.MaxTokens != 100 {
		t.Errorf("MaxTokens = %d, want 100", opts.MaxTokens)
	}
	if opts.Temperature != 0.3 {
		t.Errorf("Temperature = %f, want 0.3", opts.Temperature)
	}
}

// TestCollectSystemMessages verifies system messages are joined.
func TestCollectSystemMessages(t *testing.T) {
	prompt := fantasy.Prompt{
		fantasy.NewSystemMessage("You are helpful."),
		fantasy.NewUserMessage("Hi"),
		fantasy.NewSystemMessage("Be concise."),
	}
	got := collectSystemMessages(prompt)
	if got != "You are helpful.\n\nBe concise." {
		t.Errorf("collectSystemMessages() = %q", got)
	}
}

// TestMessageText verifies text-part concatenation.
func TestMessageText(t *testing.T) {
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "Hello, "},
			fantasy.TextPart{Text: "world"},
		},
	}
	if got := messageText(msg); got != "Hello, world" {
		t.Errorf("messageText() = %q, want %q", got, "Hello, world")
	}
}

// TestWithLogger verifies the logger option is captured by the provider.
func TestWithLogger(t *testing.T) {
	called := false
	logger := func(_ context.Context, _ string, _ ...any) { called = true }

	p, err := New(WithLogger(logger))
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	internal := p.(*provider)
	if internal.options.logger == nil {
		t.Fatal("WithLogger did not store the logger")
	}
	internal.options.logger(context.Background(), "ping")
	if !called {
		t.Error("stored logger was not the one passed in")
	}
}

// TestAdaptLogger verifies the ds4-go log callback signature is bridged
// into the fantasy Logger with the libds4 category as a structured attr.
func TestAdaptLogger(t *testing.T) {
	type call struct {
		msg  string
		args []any
	}
	var got call
	logger := func(_ context.Context, msg string, args ...any) {
		got = call{msg: msg, args: args}
	}

	adaptLogger(logger)(ds4api.LogWarning, "ctx full")

	if got.msg != "ctx full" {
		t.Errorf("msg = %q, want %q", got.msg, "ctx full")
	}
	if len(got.args) != 2 || got.args[0] != "log_type" || got.args[1] != "warning" {
		t.Errorf("args = %v, want [log_type warning]", got.args)
	}
}

// TestLogTypeName spot-checks the libds4 LogType → string mapping.
func TestLogTypeName(t *testing.T) {
	tests := map[ds4api.LogType]string{
		ds4api.LogDefault:    "default",
		ds4api.LogPrefill:    "prefill",
		ds4api.LogGeneration: "generation",
		ds4api.LogKVCache:    "kvcache",
		ds4api.LogTool:       "tool",
		ds4api.LogWarning:    "warning",
		ds4api.LogTiming:     "timing",
		ds4api.LogOK:         "ok",
		ds4api.LogError:      "error",
	}
	for typ, want := range tests {
		if got := logTypeName(typ); got != want {
			t.Errorf("logTypeName(%v) = %q, want %q", typ, got, want)
		}
	}
}

// TestDsmlTools verifies fantasy function tools convert to dsml.Tool
// definitions and non-function inputs are skipped.
func TestDsmlTools(t *testing.T) {
	if got, err := dsmlTools(nil); err != nil || len(got) != 0 {
		t.Errorf("dsmlTools(nil) = %v, %v; want empty, nil", got, err)
	}

	tools := []fantasy.Tool{
		fantasy.FunctionTool{
			Name:        "get_weather",
			Description: "Get the current weather",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
				},
			},
		},
	}
	got, err := dsmlTools(tools)
	if err != nil {
		t.Fatalf("dsmlTools() error: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("dsmlTools() len = %d, want 1", len(got))
	}
	if got[0].Name != "get_weather" {
		t.Errorf("tool name = %q, want %q", got[0].Name, "get_weather")
	}
	if len(got[0].Parameters) == 0 {
		t.Error("tool parameters not marshaled")
	}
}

// TestChatHistory verifies fantasy messages convert to ds4.ChatMessages,
// with system messages skipped and tool results expanded per part.
func TestChatHistory(t *testing.T) {
	prompt := fantasy.Prompt{
		fantasy.NewSystemMessage("You are helpful."),
		fantasy.NewUserMessage("Weather in Tokyo?"),
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "Checking."},
				fantasy.ToolCallPart{
					ToolCallID: "call-1",
					ToolName:   "get_weather",
					Input:      `{"location": "Tokyo"}`,
				},
			},
		},
		{
			Role: fantasy.MessageRoleTool,
			Content: []fantasy.MessagePart{
				fantasy.ToolResultPart{
					ToolCallID: "call-1",
					Output:     fantasy.ToolResultOutputContentText{Text: "sunny, 22C"},
				},
			},
		},
	}

	got := chatHistory(prompt)
	if len(got) != 3 {
		t.Fatalf("chatHistory() len = %d, want 3 (system skipped)", len(got))
	}

	if got[0].Role != "user" || got[0].Content != "Weather in Tokyo?" {
		t.Errorf("message 0 = %+v", got[0])
	}

	if got[1].Role != "assistant" || got[1].Content != "Checking." {
		t.Errorf("message 1 = %+v", got[1])
	}
	if len(got[1].ToolCalls) != 1 {
		t.Fatalf("assistant tool calls = %d, want 1", len(got[1].ToolCalls))
	}
	if got[1].ToolCalls[0].Name != "get_weather" || got[1].ToolCalls[0].Arguments != `{"location": "Tokyo"}` {
		t.Errorf("assistant tool call = %+v", got[1].ToolCalls[0])
	}

	if got[2].Role != "tool" || got[2].Content != "sunny, 22C" || got[2].ToolCallID != "call-1" {
		t.Errorf("message 2 = %+v", got[2])
	}
}

// collectParts runs the given events through a streamEmitter and returns
// the fantasy stream parts produced.
func collectParts(events ...dsml.StreamEvent) []fantasy.StreamPart {
	var parts []fantasy.StreamPart
	em := newStreamEmitter(func(p fantasy.StreamPart) bool {
		parts = append(parts, p)
		return true
	})
	for _, ev := range events {
		em.event(ev)
	}
	em.finish()
	return parts
}

func partTypes(parts []fantasy.StreamPart) []fantasy.StreamPartType {
	types := make([]fantasy.StreamPartType, len(parts))
	for i, p := range parts {
		types[i] = p.Type
	}
	return types
}

func deltaOf(parts []fantasy.StreamPart, typ fantasy.StreamPartType) string {
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == typ {
			sb.WriteString(p.Delta)
		}
	}
	return sb.String()
}

// TestStreamEmitter verifies dsml events translate into correctly
// bracketed fantasy stream parts.
func TestStreamEmitter(t *testing.T) {
	parts := collectParts(
		dsml.StreamEvent{Type: dsml.EventReasoningDelta, Delta: "thinking"},
		dsml.StreamEvent{Type: dsml.EventContentDelta, Delta: "Hello"},
		dsml.StreamEvent{Type: dsml.EventToolCallStart, Index: 0, Name: "get_weather"},
		dsml.StreamEvent{Type: dsml.EventToolCallArgumentsDelta, Index: 0, Delta: `{"location": "Tokyo"}`},
		dsml.StreamEvent{Type: dsml.EventToolCallEnd, Index: 0, Arguments: `{"location": "Tokyo"}`},
	)

	want := []fantasy.StreamPartType{
		fantasy.StreamPartTypeReasoningStart,
		fantasy.StreamPartTypeReasoningDelta,
		fantasy.StreamPartTypeReasoningEnd,
		fantasy.StreamPartTypeTextStart,
		fantasy.StreamPartTypeTextDelta,
		fantasy.StreamPartTypeTextEnd,
		fantasy.StreamPartTypeToolInputStart,
		fantasy.StreamPartTypeToolInputDelta,
		fantasy.StreamPartTypeToolInputEnd,
		fantasy.StreamPartTypeToolCall,
	}
	got := partTypes(parts)
	if len(got) != len(want) {
		t.Fatalf("part types = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("part %d type = %q, want %q", i, got[i], want[i])
		}
	}

	tc := parts[len(parts)-1]
	if tc.ToolCallName != "get_weather" {
		t.Errorf("ToolCall name = %q", tc.ToolCallName)
	}
	if tc.ToolCallInput != `{"location": "Tokyo"}` {
		t.Errorf("ToolCall input = %q", tc.ToolCallInput)
	}
}

// TestStreamDecoderIntegration drives a real dsml.StreamDecoder through
// the streamEmitter, verifying end-to-end translation of a completion
// with response text and a tool call.
func TestStreamDecoderIntegration(t *testing.T) {
	block, err := dsml.RenderToolCalls([]dsml.ToolCall{
		{Name: "get_weather", Arguments: `{"location": "Tokyo"}`},
	})
	if err != nil {
		t.Fatalf("RenderToolCalls() error: %v", err)
	}
	completion := "Let me check the weather." + block

	decoder := dsml.NewStreamDecoder(false)
	var parts []fantasy.StreamPart
	em := newStreamEmitter(func(p fantasy.StreamPart) bool {
		parts = append(parts, p)
		return true
	})

	// Feed the completion in small chunks to exercise chunk boundaries.
	for i := 0; i < len(completion); i += 5 {
		for _, ev := range decoder.Write(completion[i:min(i+5, len(completion))]) {
			em.event(ev)
		}
	}
	finalEvents, parsed, err := decoder.Close()
	if err != nil {
		t.Fatalf("decoder.Close() error: %v", err)
	}
	for _, ev := range finalEvents {
		em.event(ev)
	}
	em.finish()

	if got := deltaOf(parts, fantasy.StreamPartTypeTextDelta); got != "Let me check the weather." {
		t.Errorf("streamed text = %q", got)
	}
	if len(parsed.ToolCalls) != 1 {
		t.Fatalf("parsed tool calls = %d, want 1", len(parsed.ToolCalls))
	}

	var toolCall *fantasy.StreamPart
	for i := range parts {
		if parts[i].Type == fantasy.StreamPartTypeToolCall {
			toolCall = &parts[i]
		}
	}
	if toolCall == nil {
		t.Fatal("no ToolCall part emitted")
	}
	if toolCall.ToolCallName != "get_weather" {
		t.Errorf("ToolCall name = %q", toolCall.ToolCallName)
	}
	if !strings.Contains(toolCall.ToolCallInput, "Tokyo") {
		t.Errorf("ToolCall input = %q", toolCall.ToolCallInput)
	}
}

// TestStreamDecoderIntegrationReasoning verifies reasoning content streams
// as reasoning parts and the response as text parts.
func TestStreamDecoderIntegrationReasoning(t *testing.T) {
	completion := "Let me work through this</think>The answer is 42."

	decoder := dsml.NewStreamDecoder(true)
	var parts []fantasy.StreamPart
	em := newStreamEmitter(func(p fantasy.StreamPart) bool {
		parts = append(parts, p)
		return true
	})

	for _, ev := range decoder.Write(completion) {
		em.event(ev)
	}
	finalEvents, _, err := decoder.Close()
	if err != nil {
		t.Fatalf("decoder.Close() error: %v", err)
	}
	for _, ev := range finalEvents {
		em.event(ev)
	}
	em.finish()

	if got := deltaOf(parts, fantasy.StreamPartTypeReasoningDelta); got != "Let me work through this" {
		t.Errorf("streamed reasoning = %q", got)
	}
	if got := deltaOf(parts, fantasy.StreamPartTypeTextDelta); got != "The answer is 42." {
		t.Errorf("streamed text = %q", got)
	}
}

// TestProviderNewWithOptions verifies that WithPowerPercent and WithQuality options are properly captured.
func TestProviderNewWithOptions(t *testing.T) {
	p, err := New(
		WithPowerPercent(50),
		WithQuality(true),
	)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	internal := p.(*provider)
	if internal.options.powerPercent != 50 {
		t.Errorf("powerPercent = %d, want 50", internal.options.powerPercent)
	}
	if !internal.options.quality {
		t.Errorf("quality = %v, want true", internal.options.quality)
	}
}
