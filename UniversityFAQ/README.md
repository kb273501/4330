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

## 评估
data/tests.json 保存测试问题及正确 FAQ ID。
实验页面提供检索指标和 CSV 下载。
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
All demonstration policies are fictional.
