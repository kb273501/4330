# University FAQ — 双语大学问答软件

## 环境
推荐 Python 3.11。
默认使用 TF-IDF + Retrieval-only，不需要 Ollama。

## Windows
先运行 setup_windows.bat，再运行 start_windows.bat。

## macOS / Linux
先运行 bash setup_unix.sh，再运行 bash start_unix.sh。
如果默认 Python 版本过新，建议手动使用 Python 3.11 创建虚拟环境。

## 浏览器地址
http://localhost:8501

## 双语功能
中文输入默认中文回答，英文输入默认英文回答。
中英混合输入含中文时默认中文，也可以手动选择回答语言。

## Dense / Hybrid
先安装 requirements-dense.txt。
首次使用会下载多语言 embedding 模型。

## RAG
另行安装 Ollama，并执行：
ollama pull qwen2.5:3b
确保 Ollama 服务运行，然后在网页选择 RAG。

## 知识库
data/faq.json 保存双语 FAQ。
每条记录包含 id、q_zh、q_en、a_zh、a_en、source。
示例政策均为虚构，请替换为经过核实的学校资料。
网页上传只影响当前会话；永久修改请替换磁盘文件。

## 上传文档

在侧边栏上传 PDF、Word（`.docx`）或 PowerPoint（`.pptx`），每个文件最多 10 MB，最多处理前 10 个文件。提取出的文字会与 FAQ 一起参与检索；PDF 标注页码，PPT 标注幻灯片编号，Word 显示文件名。上传内容只保留在当前会话，不会写入或导出到 FAQ JSON。

直接检索模式显示文档原文，不自动翻译或总结；RAG 模式可使用本地 Ollama 根据文档生成回答。中英文关键词检索通常需要与原文语言一致。

仅提取可选择的文字，不识别扫描件或图片中的文字；不支持加密 PDF 和旧版 `.doc`、`.ppt`。如已有虚拟环境，请重新运行对应的 setup 脚本安装新增读取库。

## 评估
data/tests.json 保存测试问题及正确 FAQ ID。
实验页面提供检索指标和 CSV 下载。
评估仍只使用 FAQ，不包含上传的文档。
演示数据不能支持正式研究结论。

## 限制
当前为单轮问答，不使用历史记录推理。
未实现微调意图分类器或正式拒答校准。
RAG 可能出现事实错误，引用检查不等于事实验证。
默认仅监听本机，不适合直接公开部署。

## English
Install Python 3.11, then run the setup and start scripts.
Open http://localhost:8501 and select the English interface.
Dense / Hybrid requires optional dependencies.
RAG requires a separately installed Ollama service.
Upload PDF, DOCX, and PPTX documents in the sidebar to search their extracted text alongside the FAQs. Each file is limited to 10 MB and the first 10 files are processed. Documents stay in the current session and are not exported to FAQ JSON. Retrieval-only shows the original text; OCR, encrypted PDFs, and legacy DOC/PPT are not supported. The evaluation still uses only the FAQ. Re-run setup to install the document reader dependencies.
All demonstration policies are fictional.
