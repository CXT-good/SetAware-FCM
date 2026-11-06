# Third-party notices

## Set Transformer attention blocks

The MAB, SAB and ISAB blocks in `src/common/set_attention.py` follow the implementation in [juho-lee/set_transformer](https://github.com/juho-lee/set_transformer/blob/master/modules.py). The corresponding upstream license is retained below. Project-specific model extensions and other project files are covered by the project license unless stated otherwise.

Upstream: https://github.com/juho-lee/set_transformer

License source: https://github.com/juho-lee/set_transformer/blob/master/LICENSE

```text
MIT License

Copyright (c) 2020 Juho Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Separately installed dependencies

PyTorch, NumPy, pandas, scikit-learn, SciPy, Matplotlib, XGBoost, UMAP and other dependencies are installed separately and retain their respective licenses. The project license does not replace those terms.

## Data and checkpoints

Clinical data, patient identifiers, original split workbooks, externally obtained datasets and trained checkpoints are not licensed by this software notice. Their owners' access and distribution terms apply separately. This release does not bundle restricted clinical data or model weights.
