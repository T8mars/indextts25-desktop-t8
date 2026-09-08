# IndexTTS 2.5 本地 API 服务

Desktop 0.26.1 起内置长期稳定的本地 HTTP API。它不是某个前端的专用适配器：SillyTavern、直播工具、剪辑脚本、浏览器应用和其他程序都可以调用。API 与桌面 WebUI 共用同一模型、音色库、输出目录和 GPU 推理锁，不会为了 API 再加载一份模型。

## 最省事的启动方式

1. 首次使用先打开桌面程序，选择完整模型目录，并在“角色音色库”保存至少一个音色。
2. 在启动器展开“本地 API 服务（OpenAI 兼容）”，填写固定端口和可选默认音色，点击“保存 API 设置”。
3. 选择以下任一方式：
   - “启动 IndexTTS 2.5（同时启用 API）”：打开桌面界面，API 同时可用。
   - “仅启动 API（留在本页）”：模型常驻，但不进入 WebUI。
   - 双击便携包 EXE 同级的 `启动API服务.cmd`：独立前台控制台持续监听，日志和异常不会一闪而过。
4. `查看API服务状态.cmd` 用于检查服务，`停止API服务.cmd` 用于安全停止。前台服务也可用 `Ctrl+C` 停止。

默认地址为 `http://127.0.0.1:7861`，交互式接口文档位于 `http://127.0.0.1:7861/docs`。端口固定，不会每次启动随机变化。

## 鉴权与音色

除 `/health` 外，所有接口都要求启动器自动生成的 API Key：

```text
Authorization: Bearer <API Key>
```

也可使用 `X-API-Key: <API Key>`。Key 保存在当前 Windows 用户的桌面设置中，可以在启动器一键复制。

`voice` 只接受桌面“角色音色库”中已经保存的名称或 ID。API 不接受调用者传入任意本地音频路径，避免第三方程序借接口读取用户文件。省略 `voice` 时使用启动器配置的默认音色；若没有默认音色，请求会返回可读错误和当前可用音色。

少数 OpenAI 客户端会固定发送 `alloy / echo / nova / onyx / shimmer` 等名称。如果音色库中没有这个同名角色，但启动器已经设置默认音色，服务会自动把这些标准名称映射到本地默认音色。

## OpenAI 兼容接口

接口：`POST /v1/audio/speech`

```powershell
$headers = @{ Authorization = "Bearer <API Key>" }
$body = @{
  model = "tts-1"
  input = "1939年，这是一段 API 语音。"
  voice = "旁白"
  response_format = "mp3"
  speed = 1.0
} | ConvertTo-Json

Invoke-WebRequest `
  -Uri "http://127.0.0.1:7861/v1/audio/speech" `
  -Method Post -Headers $headers -ContentType "application/json" `
  -Body $body -OutFile "speech.mp3"
```

支持 `model=indextts-2.5 / tts-1 / tts-1-hd`，输出格式为 `mp3 / wav / pcm`，`speed` 范围为 `0.5–2.0`。可选 `language` 为 `ZH / EN / JA / ES / AR`；省略时继承音色语言。可选 `instructions` 会作为情感文本使用，但低显存模式未加载 QwenEmotion 时会明确报错。

所有成功生成的完整 WAV 默认保存在桌面输出目录，并记录到“生成历史”；HTTP 响应则按请求格式返回。可在启动器关闭历史记录，但输出文件仍会保留。

### SillyTavern / OpenAI 兼容客户端填写方式

在支持自定义 OpenAI TTS 地址的客户端中填写：

- Base URL：`http://127.0.0.1:7861/v1`
- Speech endpoint：`/audio/speech`（如果客户端要求完整地址，则填 `http://127.0.0.1:7861/v1/audio/speech`）
- API Key：启动器中复制的 Key
- Model：`tts-1` 或 `indextts-2.5`
- Voice：桌面音色库名称；客户端只能选择 `alloy` 等固定名称时，先在启动器配置本地默认音色

客户端和服务在同一台电脑时无需局域网模式，也无需填写 CORS。

## IndexTTS 原生高级接口

接口：`POST /api/v1/speech`

该接口在兼容字段之上支持 `duration_factor`、逐句情感模式、八维向量、种子、采样参数、分段长度、段间静音、文本归一化、扩散步数、CFG、CFM 温度和音频后处理。例如：

```json
{
  "model": "indextts-2.5",
  "input": "请使用平静、从容的语气介绍这项功能。",
  "voice": "旁白",
  "language": "ZH",
  "response_format": "wav",
  "emotion_mode": "vector",
  "emotion_vector": [0, 0, 0, 0, 0, 0, 0.2, 0.8],
  "emotion_strength": 0.75,
  "seed": 42,
  "text_normalization": true
}
```

八维顺序固定为：喜、怒、哀、惧、厌恶、低落、惊喜、平静。

## 异步长任务

- `POST /api/v1/jobs`：提交与原生接口相同的 JSON，立即返回 `job_id`。
- `GET /api/v1/jobs/{job_id}`：读取排队、运行、分块进度、完成或失败状态。
- `DELETE /api/v1/jobs/{job_id}`：取消等待任务；正在推理的任务会在安全分块边界停止。
- `GET /api/v1/jobs/{job_id}/audio`：任务完成后下载 WAV。

API 长任务使用单工作线程；桌面和 API 的模型调用还会经过同一全局推理锁，因此多个调用者不会同时进入 GPU 导致爆显存。

## 服务与发现接口

- `GET /health`：无需鉴权，返回版本、模型是否已加载、推理队列和任务统计。
- `GET /v1/models`：模型列表。
- `GET /v1/audio/voices`：音色名称、ID、语言和默认音色。
- `POST /api/v1/admin/shutdown`：使用 API Key 安全停止服务。

错误统一返回：

```json
{"error":{"message":"具体原因","type":"validation_error","code":"validation_error"}}
```

## 局域网与浏览器调用

默认 `127.0.0.1` 只允许本机访问，普通桌面客户端和 SillyTavern 无需开启 CORS。需要让同一局域网设备访问时，在启动器勾选“允许局域网访问”；此时监听 `0.0.0.0`，客户端应使用本机实际局域网 IP。只在可信网络使用并妥善保管 API Key。

只有浏览器页面跨域调用时才填写 CORS Origin，例如 `http://127.0.0.1:8000`。不要为了省事在不可信环境填 `*`。

## 常见问题

- `401`：没有携带 Key，或 Key 与当前服务实例不一致。
- `Unknown voice`：先在桌面音色库保存音色，或把 `voice` 改为 `/v1/audio/voices` 返回的名称/ID。
- 端口被占用：在启动器更换端口并保存；独立 CMD 与桌面会自动共用新端口。
- 启动很久：首次加载 10GB 模型会花时间，CMD 窗口会持续显示加载日志。
- 桌面与 CMD 同时启动：如果端口、Key 对应的是同一个服务，桌面会连接已有实例而不会重复加载模型。
