package ds4

import (
	"encoding/json"

	"charm.land/fantasy"
	"github.com/NimbleMarkets/ds4go/ds4api"
)

// Global type identifiers for DS4-specific provider data.
const (
	TypeProviderOptions = Name + ".options"
)

func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ProviderOptions represents additional per-call options for the DS4 provider.
type ProviderOptions struct {
	// ThinkMode controls the thinking/reasoning mode for this call.
	// When nil, the provider-level default is used.
	ThinkMode *ds4api.ThinkMode `json:"think_mode"`
	// MaxTokens overrides the call-level MaxOutputTokens for ds4.
	MaxTokens *int32 `json:"max_tokens"`
	// Temperature overrides the call-level Temperature for ds4 sampling.
	Temperature *float32 `json:"temperature"`
	// TopK overrides the call-level TopK for ds4 sampling.
	TopK *int32 `json:"top_k"`
	// TopP overrides the call-level TopP for ds4 sampling.
	TopP *float32 `json:"top_p"`
	// MinP sets ds4's minimum relative-probability filter.
	MinP *float32 `json:"min_p"`
	// Seed seeds ds4's sampler for deterministic output.
	Seed *uint64 `json:"seed"`
	// CtxSize overrides the session context window size.
	CtxSize *int `json:"ctx_size"`
}

// Options implements the ProviderOptions interface.
func (o *ProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info.
func (o ProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderOptions
	return fantasy.MarshalProviderType(TypeProviderOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info.
func (o *ProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ProviderOptions(p)
	return nil
}
