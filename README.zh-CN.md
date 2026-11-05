# 代码阅读与使用指南

本目录是论文配套代码的独立发布副本，包含 MAE 预训练、CTransformer/PACTransformer 细胞分类、CA-MIL/TACA-MIL 患者分类，以及对应基线。临床数据、患者划分、训练权重和生成结果由数据或模型所有者另行提供。

## 安装与快速检查

论文环境使用 Python 3.9.25、PyTorch 2.8.0。使用该 Python 版本创建独立环境后，在本目录运行：

```bash
python -m pip install --upgrade pip==25.2
python -m pip install -r requirements/paper-cu128.txt
python -m pip install --no-deps -e .
python examples/synthetic_demo.py
python examples/make_synthetic_data.py --output-dir outputs/synthetic
python -m pytest
```

CPU 检查将依赖文件替换为 `requirements/cpu.txt`。固定环境的版本来源及 CUDA/CPU 安装区别见 [环境说明](docs/environment.md)。合成示例使用随机权重，检查五个主要模型的 CPU 前向传播和数据接口，不代表论文性能。MAE 完整预训练要求至少两张 CUDA GPU；PACTransformer 加权训练和正式评估均支持 CPU 回退。实际检查环境与结果见 [验证记录](docs/validation.md)。

## 目录与阅读顺序

1. `src/dataio/`：marker 词表、类别映射、CSV 和患者多管数据结构。
2. `src/mae/model.py`：marker/value token、掩码重建和 MAE encoder。
3. `src/cell_level/ctransformer/model.py`：CTransformer 细胞表示与分类。
4. `src/cell_level/pactransformer/model.py`：PACTransformer 管内群体建模。
5. `src/patient_level/ca_mil/model.py` 和 `src/patient_level/taca_mil/model.py`：患者汇聚与管内群体建模。
6. `src/common/`：共享注意力与患者基线辅助模块。
7. 各模型目录的 `train.py`、`eval.py`：训练与评估；基线位于 `cell_level/baselines/` 和 `patient_level/baselines/`，跨中心辅助流程位于 `cell_level/multicenter/`。

安装后直接使用顶层包命令，无需手动修改 Python 搜索路径：

| 模型 | 训练入口 | 评估入口 |
| --- | --- | --- |
| MAE | `python -m mae.train` | 监督微调 `mae.finetune` 后用 `python -m mae.eval` |
| CTransformer | `python -m cell_level.ctransformer.train` | `python -m cell_level.ctransformer.eval` |
| PACTransformer | `python -m cell_level.pactransformer.train` | `python -m cell_level.pactransformer.eval` |
| CA-MIL | `python -m patient_level.ca_mil.train` | `python -m patient_level.ca_mil.eval` |
| TACA-MIL | `python -m patient_level.taca_mil.train` | `python -m patient_level.taca_mil.eval` |

添加 `--help` 查看参数。完整布局和模型数据流见 [阅读指南](docs/reading_guide.md)。

## 运行配置与实验记录

```bash
python scripts/run_experiment.py configs/patient_taca_mil.example.json --dry-run
```

`configs/*.example.json` 是可修改的运行模板。论文参数、找回的历史评估设置和仍缺失的运行记录见 [论文配置](docs/paper_runs.md)。执行前设置模板需要的 `FCM_DATA_ROOT`、`FCM_OUTPUT_ROOT` 和 checkpoint 环境变量，并确认参数。字段与 patient-disjoint 划分见 [数据格式](docs/data_format.md)，预处理边界及划分检查见 [预处理说明](docs/preprocessing.md)，训练、权重和评估策略见 [训练说明](docs/training.md)。运行 `python scripts/validate_manifests.py --help` 查看本地清单校验入口。

患者示例显式使用独立 `validation` sheet；历史训练器默认使用 `test` 选择最佳 checkpoint。示例不能证明历史实验采用了新的验证集协议。使用 `configs/experiment_manifest.template.csv` 记录实际划分、预处理、随机种子、权重 epoch、选择规则与评估命令，软件检查通过不等于复现临床结果。

权重迁移应使用匹配模型配置的 `state_dict` checkpoint；目录调整不保证历史整模型 pickle 可直接加载。原始研究目录不属于此发布包。

## 发布说明

v1.0.0 采用 MIT 许可证，并修复 [问题记录](docs/known_issues.md) 中的八类实现缺陷，包括 LR 解码、缺失划分回退、采样确定性、显式管型、缓存失效、PAC AP 点估计、DeepCyTOF epoch 配置和路径映射。修复会改变受影响路径的结果，历史临床指标仍需用原始数据和权重重新评估。

本目录不包含临床原始数据、患者清单、权重、预测输出、缓存或日志。代码许可见 `LICENSE`，第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，软件引用见 `CITATION.cff`。引用时记录论文 release 的 tag 和确切 commit；代码许可不授予临床数据或外部权重的使用权。详见 [发布说明](docs/publication.md)。
