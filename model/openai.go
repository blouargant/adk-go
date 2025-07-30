// Copyright 2025 The Go A2A Authors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"os"
	"runtime"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared/constant"
	"google.golang.org/genai"

	adk "github.com/go-a2a/adk-go"
	"github.com/go-a2a/adk-go/types"
)

const (
	// OpenAIDefaultModel is the default model name for OpenAI.
	OpenAIDefaultModel = "gpt-4o"

	// EnvOpenAIAPIKey is the environment variable name for the OpenAI API key.
	EnvOpenAIAPIKey = "OPENAI_API_KEY"

	// EnvOpenAIBaseURL is the environment variable name for the OpenAI base URL.
	EnvOpenAIBaseURL = "OPENAI_BASE_URL"
)

// OpenAI represents an OpenAI Large Language Model.
type OpenAI struct {
	*BaseLLM

	client openai.Client
}

var _ types.Model = (*OpenAI)(nil)

// NewOpenAI creates a new [OpenAI] instance.
func NewOpenAI(ctx context.Context, apiKey, baseURL, modelName string, opts ...Option) (*OpenAI, error) {
	// Use default model if none provided
	if modelName == "" {
		modelName = OpenAIDefaultModel
	}

	// Check API key and use [EnvOpenAIAPIKey] environment variable if not provided
	if apiKey == "" {
		envApiKey := os.Getenv(EnvOpenAIAPIKey)
		if envApiKey == "" {
			return nil, fmt.Errorf("either apiKey arg or %q environment variable must be set", EnvOpenAIAPIKey)
		}
		apiKey = envApiKey
	}

	// Setup client options
	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	// Check base URL and use environment variable if not provided
	if baseURL == "" {
		baseURL = os.Getenv(EnvOpenAIBaseURL)
	}
	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
	}

	// Add user agent headers
	frameworkLabel := fmt.Sprintf("go-a2a/adk-go/%s", adk.Version)
	languageLabel := fmt.Sprintf("go/%s", runtime.Version())
	userAgent := frameworkLabel + " " + languageLabel
	clientOpts = append(clientOpts, option.WithHeader("User-Agent", userAgent))

	// Create OpenAI client
	client := openai.NewClient(clientOpts...)

	openaiClient := &OpenAI{
		BaseLLM: NewBaseLLM(modelName, opts...),
		client:  client,
	}

	return openaiClient, nil
}

// Name returns the name of the [OpenAI] model.
func (m *OpenAI) Name() string {
	return m.modelName
}

// SupportedModels returns a list of supported models in OpenAI.
//
// See https://platform.openai.com/docs/models.
func (m *OpenAI) SupportedModels() []string {
	return []string{
		"gpt-4o",
		"gpt-4o-mini",
		"gpt-4-turbo",
		"gpt-4",
		"gpt-3.5-turbo",
		"o1-preview",
		"o1-mini",
		"o3-mini",
	}
}

// Connect creates a live connection to the OpenAI LLM.
// Note: OpenAI doesn't support live connections, so this returns an error.
func (m *OpenAI) Connect(ctx context.Context, _ *types.LLMRequest) (types.ModelConnection, error) {
	return nil, types.NotImplementedError("OpenAI: Live connection is not supported")
}

// GenerateContent generates content from the model.
func (m *OpenAI) GenerateContent(ctx context.Context, request *types.LLMRequest) (*types.LLMResponse, error) {
	// Convert genai contents to OpenAI messages
	messages, err := m.convertContentsToMessages(request.Contents)
	if err != nil {
		return nil, fmt.Errorf("convert contents to messages: %w", err)
	}

	// Build chat completion request
	chatRequest := openai.ChatCompletionNewParams{
		Model:    m.modelName,
		Messages: messages,
	}

	// Add generation config if provided
	if request.Config != nil {
		if request.Config.Temperature != nil {
			chatRequest.Temperature = openai.Float(float64(*request.Config.Temperature))
		}
		if request.Config.MaxOutputTokens > 0 {
			chatRequest.MaxTokens = openai.Int(int64(request.Config.MaxOutputTokens))
		}
		if request.Config.TopP != nil {
			chatRequest.TopP = openai.Float(float64(*request.Config.TopP))
		}
	}

	// Add tools if provided
	if request.Config != nil && len(request.Config.Tools) > 0 {
		tools, err := m.convertGenaiToolsToOpenAI(request.Config.Tools)
		if err != nil {
			return nil, fmt.Errorf("convert tools: %w", err)
		}
		chatRequest.Tools = tools
	}

	// Make the API call
	response, err := m.client.Chat.Completions.New(ctx, chatRequest)
	if err != nil {
		return nil, fmt.Errorf("openai API error: %w", err)
	}

	m.logger.DebugContext(ctx, "response", m.buildResponseLog(response))

	// Convert response to LLMResponse
	return m.convertResponseToLLMResponse(response)
}

// StreamGenerateContent streams generated content from the model.
func (m *OpenAI) StreamGenerateContent(ctx context.Context, request *types.LLMRequest) iter.Seq2[*types.LLMResponse, error] {
	return func(yield func(*types.LLMResponse, error) bool) {
		// Convert genai contents to OpenAI messages
		messages, err := m.convertContentsToMessages(request.Contents)
		if err != nil {
			yield(nil, fmt.Errorf("convert contents to messages: %w", err))
			return
		}

		// Build chat completion request
		chatRequest := openai.ChatCompletionNewParams{
			Model:    m.modelName,
			Messages: messages,
		}

		// Add generation config if provided
		if request.Config != nil {
			if request.Config.Temperature != nil {
				chatRequest.Temperature = openai.Float(float64(*request.Config.Temperature))
			}
			if request.Config.MaxOutputTokens > 0 {
				chatRequest.MaxTokens = openai.Int(int64(request.Config.MaxOutputTokens))
			}
			if request.Config.TopP != nil {
				chatRequest.TopP = openai.Float(float64(*request.Config.TopP))
			}
		}

		// Add tools if provided
		if request.Config != nil && len(request.Config.Tools) > 0 {
			tools, err := m.convertGenaiToolsToOpenAI(request.Config.Tools)
			if err != nil {
				yield(nil, fmt.Errorf("convert tools: %w", err))
				return
			}
			chatRequest.Tools = tools
		}

		// Create streaming request
		stream := m.client.Chat.Completions.NewStreaming(ctx, chatRequest)
		defer stream.Close()

		var textBuffer strings.Builder
		var lastChunk *openai.ChatCompletionChunk

		for stream.Next() {
			chunk := stream.Current()
			if ctx.Err() != nil {
				return
			}

			lastChunk = &chunk
			llmResp, err := m.convertChunkToLLMResponse(chunk)
			if err != nil {
				if !yield(nil, err) {
					return
				}
				continue
			}

			// Handle text content aggregation
			if m.containsText(llmResp) {
				textBuffer.WriteString(llmResp.Content.Parts[0].Text)
				llmResp.WithPartial(true)
			} else if textBuffer.Len() > 0 {
				// Send aggregated text before sending other content
				if !yield(m.newAggregateText(textBuffer.String()), nil) {
					return
				}
				textBuffer.Reset()
			}

			if !yield(llmResp, nil) {
				return
			}
		}

		// Check for stream errors
		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("stream error: %w", err))
			return
		}

		// Send final aggregated text if available
		if textBuffer.Len() > 0 && lastChunk != nil && m.isStreamFinished(lastChunk) {
			yield(m.newAggregateText(textBuffer.String()), nil)
		}
	}
}

// convertContentsToMessages converts genai.Content to OpenAI messages
func (m *OpenAI) convertContentsToMessages(contents []*genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(contents))

	for _, content := range contents {
		message, err := m.convertContentToMessage(content)
		if err != nil {
			return nil, fmt.Errorf("convert content to message: %w", err)
		}
		messages = append(messages, message)
	}

	return messages, nil
}

// convertContentToMessage converts a single genai.Content to OpenAI message
func (m *OpenAI) convertContentToMessage(content *genai.Content) (openai.ChatCompletionMessageParamUnion, error) {
	switch strings.ToLower(content.Role) {
	case "system":
		text := m.extractTextFromParts(content.Parts)
		return openai.SystemMessage(text), nil

	case "user":
		// Handle multimodal content
		if len(content.Parts) == 1 && content.Parts[0].Text != "" {
			// Simple text message
			return openai.UserMessage(content.Parts[0].Text), nil
		}

		// Multimodal message
		parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(content.Parts))
		for _, part := range content.Parts {
			if part.Text != "" {
				parts = append(parts, openai.TextContentPart(part.Text))
			} else if part.InlineData != nil {
				// Handle image data
				if strings.HasPrefix(part.InlineData.MIMEType, "image/") {
					imageURL := fmt.Sprintf("data:%s;base64,%s", part.InlineData.MIMEType, part.InlineData.Data)
					parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
						URL: imageURL,
					}))
				}
			}
		}

		return openai.UserMessage(parts), nil

	case "model", "assistant":
		text := m.extractTextFromParts(content.Parts)

		// Check for function calls
		var functionCalls []openai.ChatCompletionMessageToolCallParam
		for _, part := range content.Parts {
			if part.FunctionCall != nil {
				// Convert args to JSON string
				argsJSON, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("marshal function call args: %w", err)
				}

				functionCalls = append(functionCalls, openai.ChatCompletionMessageToolCallParam{
					ID:   fmt.Sprintf("call_%d", len(functionCalls)),
					Type: constant.Function("function"),
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      part.FunctionCall.Name,
						Arguments: string(argsJSON),
					},
				})
			}
		}

		assistantMsg := openai.AssistantMessage(text)
		if len(functionCalls) > 0 {
			// For function calls, we need to build it manually since the API doesn't support chaining
			return openai.ChatCompletionMessageParamOfAssistant(text), nil
		}

		return assistantMsg, nil

	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unsupported role: %s", content.Role)
	}
}

// extractTextFromParts extracts text content from genai parts
func (m *OpenAI) extractTextFromParts(parts []*genai.Part) string {
	var textParts []string
	for _, part := range parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
	}
	return strings.Join(textParts, " ")
}

// convertGenaiToolsToOpenAI converts genai tools to OpenAI tools
func (m *OpenAI) convertGenaiToolsToOpenAI(tools []*genai.Tool) ([]openai.ChatCompletionToolParam, error) {
	openaiTools := make([]openai.ChatCompletionToolParam, 0, len(tools))

	for _, tool := range tools {
		if tool.FunctionDeclarations == nil {
			continue
		}

		for _, funcDecl := range tool.FunctionDeclarations {
			openaiTool := openai.ChatCompletionToolParam{
				Type: constant.Function("function"),
				Function: openai.FunctionDefinitionParam{
					Name:        funcDecl.Name,
					Description: openai.String(funcDecl.Description),
				},
			}

			// Convert parameters schema if available
			if funcDecl.Parameters != nil {
				// Convert genai.Schema to interface{} (note: this may require custom conversion)
				params := make(map[string]any)
				// For now, we'll skip parameter conversion or implement a proper converter
				openaiTool.Function.Parameters = params
			}

			openaiTools = append(openaiTools, openaiTool)
		}
	}

	return openaiTools, nil
}

// convertResponseToLLMResponse converts OpenAI response to LLMResponse
func (m *OpenAI) convertResponseToLLMResponse(response *openai.ChatCompletion) (*types.LLMResponse, error) {
	if len(response.Choices) == 0 {
		return &types.LLMResponse{
			ErrorCode:    "NO_CHOICES",
			ErrorMessage: "OpenAI response contains no choices",
		}, nil
	}

	choice := response.Choices[0]
	content := &genai.Content{
		Role: RoleModel,
	}

	// Handle text content
	if choice.Message.Content != "" {
		content.Parts = append(content.Parts, genai.NewPartFromText(choice.Message.Content))
	}

	// Handle function calls
	for _, toolCall := range choice.Message.ToolCalls {
		if toolCall.Function.Name != "" {
			// Parse function arguments
			var args map[string]any
			if toolCall.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					// If parsing fails, use empty args
					args = map[string]any{}
				}
			}

			content.Parts = append(content.Parts, genai.NewPartFromFunctionCall(
				toolCall.Function.Name,
				args,
			))
		}
	}

	llmResponse := &types.LLMResponse{
		Content: content,
	}

	// Map finish reason
	switch choice.FinishReason {
	case "stop":
		llmResponse.FinishReason = genai.FinishReasonStop
	case "length":
		llmResponse.FinishReason = genai.FinishReasonMaxTokens
	case "content_filter":
		llmResponse.FinishReason = genai.FinishReasonSafety
	case "tool_calls":
		llmResponse.FinishReason = genai.FinishReasonStop
	}

	// Add usage metadata if available
	if response.Usage.TotalTokens > 0 {
		llmResponse.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(response.Usage.PromptTokens),
			CandidatesTokenCount: int32(response.Usage.CompletionTokens),
			TotalTokenCount:      int32(response.Usage.TotalTokens),
		}
	}

	return llmResponse, nil
}

// convertChunkToLLMResponse converts OpenAI streaming chunk to LLMResponse
func (m *OpenAI) convertChunkToLLMResponse(chunk openai.ChatCompletionChunk) (*types.LLMResponse, error) {
	if len(chunk.Choices) == 0 {
		return &types.LLMResponse{}, nil
	}

	choice := chunk.Choices[0]
	content := &genai.Content{
		Role: RoleModel,
	}

	// Handle delta content
	if choice.Delta.Content != "" {
		content.Parts = append(content.Parts, genai.NewPartFromText(choice.Delta.Content))
	}

	// Handle function calls in delta
	for _, toolCall := range choice.Delta.ToolCalls {
		if toolCall.Function.Name != "" {
			// Parse function arguments
			var args map[string]any
			if toolCall.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					// If parsing fails, use empty args
					args = map[string]any{}
				}
			}

			content.Parts = append(content.Parts, genai.NewPartFromFunctionCall(
				toolCall.Function.Name,
				args,
			))
		}
	}

	llmResponse := &types.LLMResponse{
		Content: content,
	}

	// Map finish reason
	if choice.FinishReason != "" {
		switch choice.FinishReason {
		case "stop":
			llmResponse.FinishReason = genai.FinishReasonStop
		case "length":
			llmResponse.FinishReason = genai.FinishReasonMaxTokens
		case "content_filter":
			llmResponse.FinishReason = genai.FinishReasonSafety
		case "tool_calls":
			llmResponse.FinishReason = genai.FinishReasonStop
		}
	}

	return llmResponse, nil
}

// containsText returns true when the first part has a non-empty Text field.
func (m *OpenAI) containsText(r *types.LLMResponse) bool {
	return r.Content != nil && len(r.Content.Parts) > 0 && r.Content.Parts[0].Text != ""
}

// newAggregateText creates an LLMResponse with aggregated text
func (m *OpenAI) newAggregateText(text string) *types.LLMResponse {
	return &types.LLMResponse{
		Content: &genai.Content{
			Role:  RoleModel,
			Parts: []*genai.Part{genai.NewPartFromText(text)},
		},
	}
}

// isStreamFinished checks if the stream has finished
func (m *OpenAI) isStreamFinished(chunk *openai.ChatCompletionChunk) bool {
	if len(chunk.Choices) == 0 {
		return false
	}
	return chunk.Choices[0].FinishReason == "stop"
}

// buildResponseLog builds a log entry for the response
func (m *OpenAI) buildResponseLog(response *openai.ChatCompletion) slog.Attr {
	if len(response.Choices) == 0 {
		return slog.String("response", "No choices in response")
	}

	choice := response.Choices[0]
	var functionCalls []string
	for _, toolCall := range choice.Message.ToolCalls {
		functionCalls = append(functionCalls, fmt.Sprintf("name: %s, args: %s",
			toolCall.Function.Name, toolCall.Function.Arguments))
	}

	logText := fmt.Sprintf(`
OpenAI Response:
-----------------------------------------------------------
Text:
%s
-----------------------------------------------------------
Function calls:
%s
-----------------------------------------------------------
`, choice.Message.Content, strings.Join(functionCalls, "\n"))

	return slog.String("response", logText)
}
