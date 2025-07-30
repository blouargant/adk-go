// Copyright 2025 The Go A2A Authors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"fmt"
	"iter"
	"strings"

	"google.golang.org/genai"

	"github.com/go-a2a/adk-go/types"
)

// BaseLLM represents a base LLM implementation.
type BaseLLM struct {
	Config

	// modelName represents the specific LLM model name.
	modelName string
}

var _ types.Model = (*BaseLLM)(nil)

// NewBaseLLM returns the new [BaseLLM] with the specified model name.
func NewBaseLLM(modelName string, opts ...Option) *BaseLLM {
	llm := &BaseLLM{
		Config:    newConfig(),
		modelName: modelName,
	}

	for _, opt := range opts {
		llm.Config = opt.apply(llm.Config)
	}

	return llm
}

// Name implements [Model].
func (m *BaseLLM) Name() string {
	return m.modelName
}

// SupportedModels implements [Model].
func (m *BaseLLM) SupportedModels() []string {
	return nil
}

// Connect implements [Model].
func (m *BaseLLM) Connect(context.Context, *types.LLMRequest) (types.ModelConnection, error) {
	return nil, types.NotImplementedError(fmt.Sprintf("BaseLLM: Live connection is not supported for %s", m.modelName))
}

// GenerateContent implements [Model].
func (m *BaseLLM) GenerateContent(ctx context.Context, request *types.LLMRequest) (*types.LLMResponse, error) {
	return nil, types.NotImplementedError(fmt.Sprintf("BaseLLM: async generation is not supported for %s", m.modelName))
}

// StreamGenerateContent implements [Model].
func (m *BaseLLM) StreamGenerateContent(ctx context.Context, request *types.LLMRequest) iter.Seq2[*types.LLMResponse, error] {
	return func(yield func(*types.LLMResponse, error) bool) {
		yield(nil, types.NotImplementedError(fmt.Sprintf("BaseLLM: async generation is not supported for %s", m.modelName)))
	}
}

// appendUserContent checks if the last message is from the user and if not, appends an empty user message.
func (m *BaseLLM) appendUserContent(contents []*genai.Content) []*genai.Content {
	switch {
	case len(contents) == 0:
		return append(contents, &genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				genai.NewPartFromText(`Handle the requests as specified in the System Instruction.`),
			},
		})

	case strings.ToLower(contents[len(contents)-1].Role) != genai.RoleUser:
		return append(contents, &genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				genai.NewPartFromText(`Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.`),
			},
		})

	default:
		return contents
	}
}
