// server.js - OpenAI to NVIDIA NIM API Proxy (Tailored for GLM-5.2 Max Thinking)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Must be true to pass back <think> tags to maintain chat history continuity
const SHOW_REASONING = true; 

// Base Model Map - Routes OpenAI picker names to official NVIDIA NIM targets
const MODEL_MAPPING = {
  'glm-5.2': 'z-ai/glm-5.2',
  'gpt-4o': 'z-ai/glm-5.2', // Optional shortcut: maps standard JanitorAI gpt-4o selections straight to GLM
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'minimax-m3': 'minimax/minimax-m3',
  'step-3.7': 'stepfun-ai/step-3.7-flash'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    glm_max_thinking: 'ENABLED',
    reasoning_display: SHOW_REASONING
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Fallback to GLM-5.2 if an unmapped model string is inputted
    let nimModel = MODEL_MAPPING[model] || 'z-ai/glm-5.2';
    
    // Construct base payload
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 1.0, // Z.AI recommends 1.0 for GLM-5.2 thinking tasks
      stream: stream || false
    };

    // 💡 CUSTOM INJECTION FOR GLM-5.2 MAX THINKING
    if (nimModel === 'z-ai/glm-5.2') {
      nimRequest.thinking = { type: "enabled" };
      nimRequest.reasoning_effort = "low";
      nimRequest.max_tokens = max_tokens || 4096; // Allocated buffer for prolonged thinking tokens
    } else if (nimModel.includes('thinking') || nimModel.includes('r1')) {
      // Logic fallback block for DeepSeek/Qwen thinking types
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
      nimRequest.max_tokens = max_tokens || 9024;
    } else {
      nimRequest.max_tokens = max_tokens || 4096;
    }
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                // Read incoming GLM reasoning tokens or fallback standard streams
                const reasoning = data.choices[0].delta.reasoning_content || data.choices[0].delta.reasoning;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                    delete data.choices[0].delta.reasoning;
                  }
                } else {
                  if (content) data.choices[0].delta.content = content;
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Non-streaming fallback formatting logic
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          const targetReasoning = choice.message?.reasoning_content || choice.message?.reasoning;
          
          if (SHOW_REASONING && targetReasoning) {
            fullContent = '<think>\n' + targetReasoning + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all route definition
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`GLM-5.2 Max Thinking profile initialized successfully.`);
});
