package openwakeword

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestPredictionHistoryHelpers(t *testing.T) {
	history := make([]float32, 0, predictionHistory+5)
	for i := 0; i < predictionHistory+5; i++ {
		history = appendHistory(history, float32(i))
	}
	if len(history) != predictionHistory {
		t.Fatalf("expected %d history entries, got %d", predictionHistory, len(history))
	}
	if history[0] != 5 {
		t.Fatalf("expected oldest retained score 5, got %v", history[0])
	}
	if got := countAtLeast(tail(history, 4), 32); got != 3 {
		t.Fatalf("expected 3 scores at least 32, got %d", got)
	}
}

func TestLastPositiveDimension(t *testing.T) {
	if got := lastPositiveDimension([]int64{-1, 1, 7}); got != 7 {
		t.Fatalf("expected 7, got %d", got)
	}
	if got := lastPositiveDimension([]int64{-1, -1}); got != 0 {
		t.Fatalf("expected 0, got %d", got)
	}
}

func TestDecodeWAV(t *testing.T) {
	samples := []int16{-32768, -1, 0, 1, 32767}
	var data bytes.Buffer
	data.WriteString("RIFF")
	_ = binary.Write(&data, binary.LittleEndian, uint32(36+len(samples)*2))
	data.WriteString("WAVEfmt ")
	_ = binary.Write(&data, binary.LittleEndian, uint32(16))
	_ = binary.Write(&data, binary.LittleEndian, uint16(1))
	_ = binary.Write(&data, binary.LittleEndian, uint16(1))
	_ = binary.Write(&data, binary.LittleEndian, uint32(SampleRate))
	_ = binary.Write(&data, binary.LittleEndian, uint32(SampleRate*2))
	_ = binary.Write(&data, binary.LittleEndian, uint16(2))
	_ = binary.Write(&data, binary.LittleEndian, uint16(16))
	data.WriteString("data")
	_ = binary.Write(&data, binary.LittleEndian, uint32(len(samples)*2))
	for _, sample := range samples {
		_ = binary.Write(&data, binary.LittleEndian, sample)
	}

	wav, err := DecodeWAV(bytes.NewReader(data.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(wav.Samples) != len(samples) || wav.Samples[0] != samples[0] || wav.Samples[4] != samples[4] {
		t.Fatalf("unexpected WAV samples: %+v", wav)
	}
}
