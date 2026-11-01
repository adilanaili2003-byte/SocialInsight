\# SocialInsight



\*\*基于大语言模型的社交媒体舆情分析系统\*\*



SocialInsight 是一个基于 Python 和大语言模型构建的舆情分析应用，支持上传 CSV 格式的社交媒体评论数据，自动完成数据清洗、情感分析、关键词提取、主题聚类和舆情风险识别，并结合大语言模型生成结构化的舆情分析报告。



项目采用模块化设计，将数据处理、文本分析、风险识别和 AI 报告生成等功能拆分为独立模块，通过统一工作流完成整个分析过程，并使用 Streamlit 提供可视化交互界面。



\## 一、项目功能



| 功能模块     | 功能说明                                |

| -------- | ----------------------------------- |

| 数据上传     | 支持上传 CSV 格式的评论数据                    |

| 数据清洗     | 处理空值、重复评论、文本格式及异常数据                 |

| 情感分析     | 使用 TF-IDF、逻辑回归、本地规则和 SnowNLP 进行情感判断 |

| LLM 情感精调 | 可选调用 DeepSeek 对部分评论进行进一步判断          |

| 关键词提取    | 使用 Jieba 和 TF-IDF 提取评论关键词           |

| 主题聚类     | 使用 KMeans 对评论进行主题聚类                 |

| 风险识别     | 根据风险关键词识别潜在舆情问题                     |

| AI 报告生成  | 调用 DeepSeek-V4-Pro 生成舆情分析报告         |

| 数据可视化    | 通过 Streamlit 展示分析结果                 |

| 文件导出     | 支持导出清洗数据、情感分析结果、主题分析结果和 Markdown 报告 |



\## 二、技术栈



\*\*开发语言\*\*



\* Python 3.11



\*\*数据处理\*\*



\* Pandas

\* NumPy



\*\*自然语言处理\*\*



\* Jieba

\* SnowNLP

\* Scikit-learn

\* TF-IDF

\* Logistic Regression

\* KMeans



\*\*大语言模型\*\*



\* DeepSeek-V4-Pro

\* OpenAI SDK 兼容接口



\*\*前端与可视化\*\*



\* Streamlit

\* Matplotlib

\* Seaborn

\* WordCloud



\## 三、项目架构



```text

SocialInsight/

│

├── assets/                     # 静态资源

│

├── data/

│   ├── raw/                    # 原始评论数据

│   ├── processed/              # 清洗及分析结果

│   └── models/                 # 本地情感模型

│

├── reports/                    # 分析报告

│

├── src/

│   ├── data\_processor.py       # 数据清洗

│   ├── sentiment\_analyzer.py   # 情感分析

│   ├── sentiment\_model.py      # 本地情感模型

│   ├── sentiment\_refiner.py    # 情感规则及LLM精调

│   ├── topic\_analyzer.py       # 关键词提取与主题聚类

│   ├── risk\_analyzer.py        # 舆情风险识别

│   ├── report\_generator.py     # AI报告生成

│   │

│   └── agents/

│       └── workflow.py         # 统一分析工作流

│

├── app.py                      # Streamlit应用入口

├── train\_sentiment\_model.py    # 情感模型训练脚本

├── requirements.txt            # 项目依赖

├── .env                        # 环境变量配置

└── README.md

```



\## 四、运行环境



\* Windows 10 / 11

\* Python 3.11

\* 建议使用虚拟环境管理项目依赖



\## 五、安装与运行



\### 1. 克隆项目



```bash

git clone <your-github-repository-url>

cd SocialInsight

```



\### 2. 创建虚拟环境



```bash

python -m venv .venv

```



\### 3. 安装依赖



```bash

pip install -r requirements.txt

```



\### 4. 配置环境变量



在项目根目录创建 `.env` 文件：



```env

OPENAI\_API\_KEY=your\_api\_key

OPENAI\_BASE\_URL=https://api.deepseek.com

OPENAI\_MODEL=deepseek-v4-pro

```



将 `your\_api\_key` 替换为自己的 DeepSeek API Key。



注意：不要将包含真实 API Key 的 `.env` 文件上传到 GitHub。



\### 5. 启动项目



```bash

streamlit run app.py

```



启动后，在浏览器访问：



```text

http://localhost:8501

```



\## 六、分析流程



系统采用模块化工作流组织各个分析模块，整体流程如下：



```text

&#x20;       CSV评论数据

&#x20;             │

&#x20;             ▼

&#x20;         数据读取

&#x20;             │

&#x20;             ▼

&#x20;         数据清洗

&#x20;             │

&#x20;             ▼

&#x20;         情感分析

&#x20;             │

&#x20;             ▼

&#x20;         关键词提取

&#x20;             │

&#x20;             ▼

&#x20;         主题聚类

&#x20;             │

&#x20;             ▼

&#x20;         舆情风险识别

&#x20;             │

&#x20;             ▼

&#x20;       分析结果汇总

&#x20;             │

&#x20;             ▼

&#x20;      DeepSeek生成报告

&#x20;             │

&#x20;             ▼

&#x20;      Streamlit结果展示

&#x20;             │

&#x20;             ▼

&#x20;         文件导出

```



其中，情感分析以本地模型和规则判断为主，DeepSeek 情感精调为可选功能；AI 报告生成独立于本地分析流程。



\## 七、数据格式



系统通过 CSV 文件读取评论数据，评论文本字段需要包含：



| 字段名     | 说明     |

| ------- | ------ |

| content | 评论文本内容 |



示例：



| content        |

| -------------- |

| 这家店味道不错，下次还会购买 |

| 商品质量一般，不太满意    |

| 物流速度很快，包装也很好   |



\## 八、项目说明与局限性



1\. 当前情感分类主要使用外卖评论数据训练的 TF-IDF + Logistic Regression 模型，模型在不同领域数据上的表现可能存在差异。

2\. 情感规则主要用于识别部分明确的正面、负面和中性表达，复杂语境、反讽和上下文判断仍存在局限。

3\. 当前本地情感模型采用二分类训练数据，三分类结果中的中性类别主要通过规则和概率区间判断实现。

4\. 主题聚类采用 TF-IDF + KMeans 方法，聚类结果会受到数据规模、文本质量和聚类参数影响。

5\. 舆情风险识别主要依赖关键词匹配，风险词典和等级阈值仍有进一步优化空间。

6\. 当前系统通过 CSV 上传数据，不包含社交媒体平台的实时爬取功能。



\## 九、后续优化方向



\* 优化情感分类模型，提高跨领域文本分析能力。

\* 完善复杂语境、否定句和反讽表达的识别。

\* 优化主题聚类质量和主题标签生成。

\* 扩展舆情风险词典及风险识别策略。

\* 增加更多数据可视化和分析维度。

\* 探索 Agent 工具调用与动态任务规划，实现更智能的舆情分析流程。



\## 十、项目定位



SocialInsight 是一个集数据处理、自然语言处理、机器学习和大语言模型应用于一体的 Python AI 应用项目，主要用于展示文本分析、AI 工作流设计、LLM API 集成和 Web 应用开发能力。



