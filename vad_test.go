package openwakeword

import "testing"

func TestVADContextScore(t *testing.T) {
	vad := &VAD{
		opts: VADOptions{
			ContextWindowStartOffset: defaultVADContextWindowStartOffset,
			ContextWindowEndOffset:   defaultVADContextWindowEndOffset,
		},
		history: Samples{
			0.1,
			0.2,
			0.7,
			0.4,
			0.9,
			0.1,
			0.2,
			0.3,
			0.8,
		},
	}

	score := vad.ContextScore()
	if score != 0.9 {
		t.Fatalf("ContextScore() = %v, want 0.9", score)
	}
}

func TestVADContextScoreCustomWindow(t *testing.T) {
	vad := &VAD{
		opts: VADOptions{
			ContextWindowStartOffset: 3,
			ContextWindowEndOffset:   0,
		},
		history: Samples{0.1, 0.2, 0.9, 0.3, 0.4, 0.8},
	}

	score := vad.ContextScore()
	if score != 0.8 {
		t.Fatalf("ContextScore() = %v, want 0.8", score)
	}
}

func TestVADContextScoreShortHistory(t *testing.T) {
	vad := &VAD{
		opts: VADOptions{
			ContextWindowStartOffset: defaultVADContextWindowStartOffset,
			ContextWindowEndOffset:   defaultVADContextWindowEndOffset,
		},
		history: Samples{0.9, 0.8, 0.7, 0.6, 0.5, 0.4},
	}

	score := vad.ContextScore()
	if score != 0 {
		t.Fatalf("ContextScore() = %v, want 0", score)
	}
}

func TestVADOptionsNormalize(t *testing.T) {
	opts := VADOptions{}

	if err := opts.normalize(); err != nil {
		t.Fatal(err)
	}
	if opts.FrameSize != defaultVADFrameSize {
		t.Fatalf("FrameSize = %d, want %d", opts.FrameSize, defaultVADFrameSize)
	}
	if opts.Threshold != defaultVADThreshold {
		t.Fatalf("Threshold = %v, want %v", opts.Threshold, defaultVADThreshold)
	}
	if opts.ContextWindowStartOffset != defaultVADContextWindowStartOffset {
		t.Fatalf("ContextWindowStartOffset = %d, want %d", opts.ContextWindowStartOffset, defaultVADContextWindowStartOffset)
	}
	if opts.ContextWindowEndOffset != defaultVADContextWindowEndOffset {
		t.Fatalf("ContextWindowEndOffset = %d, want %d", opts.ContextWindowEndOffset, defaultVADContextWindowEndOffset)
	}
}

func TestVADOptionsNormalizeAllowsZeroEndOffset(t *testing.T) {
	opts := VADOptions{
		ContextWindowStartOffset: 3,
		ContextWindowEndOffset:   0,
	}

	if err := opts.normalize(); err != nil {
		t.Fatal(err)
	}
	if opts.ContextWindowEndOffset != 0 {
		t.Fatalf("ContextWindowEndOffset = %d, want 0", opts.ContextWindowEndOffset)
	}
}

func TestVADOptionsNormalizeInvalidContextWindow(t *testing.T) {
	tests := []struct {
		name string
		opts VADOptions
	}{
		{
			name: "negative start offset",
			opts: VADOptions{ContextWindowStartOffset: -1},
		},
		{
			name: "negative end offset",
			opts: VADOptions{ContextWindowStartOffset: 3, ContextWindowEndOffset: -1},
		},
		{
			name: "start equals end",
			opts: VADOptions{ContextWindowStartOffset: 3, ContextWindowEndOffset: 3},
		},
		{
			name: "start before end",
			opts: VADOptions{ContextWindowStartOffset: 2, ContextWindowEndOffset: 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.opts.normalize()
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
