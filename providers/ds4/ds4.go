// Package ds4 provides an implementation of the fantasy AI SDK for
// DeepSeek V4 Flash, using the native low-level ds4 inference engine
// via github.com/NimbleMarkets/ds4go.
//
// Unlike HTTP/OpenAI-compat providers, ds4 runs inference in-process
// with zero-copy token handling and persistent KV-cache session reuse.
// This gives lower latency, richer thinking-mode support, and efficient
// multi-turn agent loops.
package ds4

import (
	"cmp"
	"context"
	"sync"

	"charm.land/fantasy"
	"github.com/NimbleMarkets/ds4go"
	"github.com/NimbleMarkets/ds4go/ds4api"
)

const (
	// Name is the name of the DS4 provider.
	Name = "ds4"

	// defaultModelID is the only model ds4 currently serves.
	defaultModelID = "deepseek-v4-flash"
)

type provider struct {
	options options

	// engine is built lazily on first generation and shared across
	// every language model and session created by this provider.
	mu          sync.Mutex
	engine      *ds4api.Engine
	engineErr   error
	engineBuilt bool

	// logOnce guards the one-time install of the libds4 log callback.
	// libds4's logger is process-global, so we install it only on the
	// first engine resolution and never replace it from here.
	logOnce sync.Once
}

// Logger is the function signature this provider uses to surface
// libds4 diagnostics to its embedder. It matches the signature used by
// other fantasy providers (see providers/kronk.Logger) so the same
// adapter — e.g. a wrapped *slog.Logger — can be reused across providers.
//
// The libds4 log category is passed as a structured attribute under the
// key "log_type". Callbacks may run on native worker threads and must be
// concurrency-safe.
type Logger func(ctx context.Context, msg string, args ...any)

type options struct {
	name             string
	engine           *ds4api.Engine
	modelPath        string
	backend          ds4api.Backend
	defaultThinkMode ds4api.ThinkMode
	nThreads         int
	mtpDraftTokens   int
	mtpMargin        float32
	warmWeights      bool
	powerPercent     int
	quality          bool
	objectMode       fantasy.ObjectMode
	logger           Logger
}

// Option defines a function that configures DS4 provider options.
type Option = func(*options)

// New creates a new DS4 provider with the given options.
// The provider runs DeepSeek V4 Flash in-process using the native
// ds4 inference engine — no HTTP round-trips, no OpenAI compat layer.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		name:             Name,
		backend:          ds4api.BackendMetal,
		defaultThinkMode: ds4api.ThinkNone,
		nThreads:         0,
		objectMode:       fantasy.ObjectModeTool,
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.name = cmp.Or(providerOptions.name, Name)

	return &provider{options: providerOptions}, nil
}

// WithName sets a custom provider name.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithEngine sets a pre-built ds4 engine to share across models and sessions.
// When set, WithModelPath and WithBackend are ignored.
func WithEngine(engine *ds4api.Engine) Option {
	return func(o *options) {
		o.engine = engine
	}
}

// WithModelPath sets the path to the DeepSeek V4 Flash GGUF model file.
func WithModelPath(modelPath string) Option {
	return func(o *options) {
		o.modelPath = modelPath
	}
}

// WithBackend selects the ds4 accelerator backend (Metal, CUDA, or CPU).
func WithBackend(backend ds4api.Backend) Option {
	return func(o *options) {
		o.backend = backend
	}
}

// WithDefaultThinkMode sets the default thinking mode used when no
// per-call think mode is provided via ProviderOptions.
func WithDefaultThinkMode(mode ds4api.ThinkMode) Option {
	return func(o *options) {
		o.defaultThinkMode = mode
	}
}

// WithNThreads sets the number of CPU worker threads.
func WithNThreads(n int) Option {
	return func(o *options) {
		o.nThreads = n
	}
}

// WithMTPDraftTokens enables speculative decoding with the given draft length.
func WithMTPDraftTokens(n int) Option {
	return func(o *options) {
		o.mtpDraftTokens = n
	}
}

// WithMTPMargin sets the speculative decoding acceptance margin.
func WithMTPMargin(margin float32) Option {
	return func(o *options) {
		o.mtpMargin = margin
	}
}

// WithWarmWeights asks ds4 to warm model weights after loading.
func WithWarmWeights(warm bool) Option {
	return func(o *options) {
		o.warmWeights = warm
	}
}

// WithPowerPercent throttles GPU work to roughly this duty cycle (1..100).
// 0 or 100 disables throttling.
func WithPowerPercent(percent int) Option {
	return func(o *options) {
		o.powerPercent = percent
	}
}

// WithQuality requests ds4's quality-oriented execution path where supported.
func WithQuality(quality bool) Option {
	return func(o *options) {
		o.quality = quality
	}
}

// WithObjectMode sets the object generation mode.
// Supported modes: ObjectModeTool, ObjectModeText.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		if om == fantasy.ObjectModeAuto || om == fantasy.ObjectModeJSON {
			om = fantasy.ObjectModeTool
		}
		o.objectMode = om
	}
}

// WithLogger routes libds4 diagnostics (prefill, generation, KV-cache,
// tool, timing, warning, error, …) into the given logger.
//
// libds4 owns one process-global log sink, so this provider installs the
// callback exactly once — on the first Generate or Stream call — and
// never replaces it. If your application uses multiple ds4 providers,
// only the first one to actually run gets to set the sink.
func WithLogger(logger Logger) Option {
	return func(o *options) {
		o.logger = logger
	}
}

// getEngine returns the shared ds4 engine, building it on first use.
// A pre-built engine supplied via WithEngine is returned as-is. On the
// first successful resolution it also installs the libds4 log callback
// when WithLogger was provided.
func (p *provider) getEngine() (*ds4api.Engine, error) {
	p.installLoggerOnce()
	engine, err := p.resolveEngine()
	if err != nil || engine == nil {
		return engine, err
	}
	return engine, nil
}

func (p *provider) resolveEngine() (*ds4api.Engine, error) {
	if p.options.engine != nil {
		return p.options.engine, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.engineBuilt {
		return p.engine, p.engineErr
	}
	p.engineBuilt = true
	p.engine, p.engineErr = ds4.NewEngine(ds4api.EngineOptions{
		ModelPath:      p.options.modelPath,
		Backend:        p.options.backend,
		NThreads:       p.options.nThreads,
		MTPDraftTokens: p.options.mtpDraftTokens,
		MTPMargin:      p.options.mtpMargin,
		WarmWeights:    p.options.warmWeights,
		PowerPercent:   p.options.powerPercent,
		Quality:        p.options.quality,
	})
	return p.engine, p.engineErr
}

func (p *provider) installLoggerOnce() {
	if p.options.logger == nil {
		return
	}
	p.logOnce.Do(func() {
		_ = ds4.SetLogFunc(adaptLogger(p.options.logger))
	})
}

// adaptLogger bridges the ds4-go log callback to a fantasy-style Logger.
// libds4 may invoke the callback from native worker threads; the user's
// Logger therefore must be concurrency-safe.
func adaptLogger(logger Logger) ds4api.LogFunc {
	return func(typ ds4api.LogType, msg string) {
		logger(context.Background(), msg, "log_type", logTypeName(typ))
	}
}

func logTypeName(typ ds4api.LogType) string {
	switch typ {
	case ds4api.LogDefault:
		return "default"
	case ds4api.LogPrefill:
		return "prefill"
	case ds4api.LogGeneration:
		return "generation"
	case ds4api.LogKVCache:
		return "kvcache"
	case ds4api.LogTool:
		return "tool"
	case ds4api.LogWarning:
		return "warning"
	case ds4api.LogTiming:
		return "timing"
	case ds4api.LogOK:
		return "ok"
	case ds4api.LogError:
		return "error"
	default:
		return "unknown"
	}
}

// LanguageModel implements fantasy.Provider. The engine is not built
// here — it is created lazily on the first Generate or Stream call so
// that constructing a model is cheap and never touches disk.
func (p *provider) LanguageModel(_ context.Context, modelID string) (fantasy.LanguageModel, error) {
	return newLanguageModel(
		p.options.name,
		cmp.Or(modelID, defaultModelID),
		p.getEngine,
		p.options.defaultThinkMode,
		p.options.objectMode,
	), nil
}

func (p *provider) Name() string {
	return p.options.name
}
