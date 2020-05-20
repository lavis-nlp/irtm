# RŶN

> But no wizardry nor spell, neither fang nor venom, nor devil's art
> nor beast-strength, could overthrow Huan without forsaking his body
> utterly.

## Installation

```bash
conda create --name ryn python=3.8
pip install -r requirements.txt
pip install -e .
```


### OpenKE Integration

Install `openke` python module (not offered by the repository itself
and thus patched by me in the most unintrusive way I found). Last
tested working version is `7a561c2`.

``` bash
pushd lib/OpenKE
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE
pushd OpenKE/openke/
bash make.sh
popd
pip install -e .
popd
```
