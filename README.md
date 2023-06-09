# probabilistic-flow-circuits
This repository contains the code for the research paper <b>Probabilistic Flow Circuits:
Towards Unified Deep Models for Tractable Probabilistic Inference</b>, which was presented at UAI 2023.

### Setup Instructions
Create a new conda or virtual environment.
Activate the environment and install  the required packages
```shell
conda activate ENV_NAME   # or source venv/bin/activate
pip install -r requirements.txt 
```

### How to run
You can use <i><b> trainer.py </b></i> to run the experiments. You can specify the model, dataset and hyperparameters using <i><b>config.py</b></i> or by passing them as command line arguments to  <i> trainer.py </i>. For example,
 
To train EinsumNet+LRS on the 3D dataset KNOTTED, you may run:
```shell
python trainer.py --dataset KNOTTED --model LinearSplineEinsumFlow   --config '{"log_freq":10, "lr":0.001, "epochs":200}' --graph random_binary_tree
```

Similarly, to train EinsumNet on the 3D dataset KNOTTED, you may run:
```shell
python trainer.py --dataset KNOTTED --model EinsumNet   --config '{"log_freq":10, "lr":0.001, "epochs":200}' --graph random_binary_tree
```

## Citation
If you find probabilistic flow circuits useful in your research, kindly cite the following paper:

```python
@inproceedings{
    sidheekh2023probabilistic,
    title={Probabilistic Flow Circuits: Towards Unified Deep Models for Tractable Probabilistic Inference},
    author={Sahil Sidheekh, Kristian Kersting, Sriraam Natarajan},
    booktitle={The 39th Conference on Uncertainty in Artificial Intelligence},
    year={2023},
    url={https://openreview.net/forum?id=1oE7YizXHf}
}
```

## License

This project is distributed under [MIT license](LICENSE).

```c
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```