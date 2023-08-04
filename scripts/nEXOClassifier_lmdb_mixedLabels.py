#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
#model code copied from https://github.com/DeepLearnPhysics/pytorch-uresnet

import numpy as np
from numpy import load
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import argparse
#from networks.preact_resnet import PreActResNet18
import sys
sys.path += '/p/lustre1/scotswas/nexo-builds/nexo-main/build/python:/p/lustre1/scotswas/nexo-builds/nexo-main/build/lib:/p/lustre1/nexouser/nexo-spack-env-v0.19/.spack-env/view/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-psutil-5.9.2-66d5raqsiow4ei4gxcnhu6hhglzqpif3/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pyparsing-3.0.9-mc2sfrh7ohajydba3c544oghbs7uim23/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-packaging-21.3-ewfblpptepo6qzfjdagfmrs2uvtlupci/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-tornado-6.2-5i5srexazblzso4mgjixoxato4ygkrsa/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-greenlet-1.1.3-csmowikxkzhbt46uibfxjbuluy7cbkmv/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pycparser-2.21-dwhn3godjbhmjgc55vwlrezyximfivsw/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-cffi-1.15.0-hqxzhfka524jtidjvxr7o7llgnju5plg/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-gevent-1.5.0-d7fld2cq4swcb6ztszymdmsmgb5m5xta/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pyzmq-24.0.1-46jv2cm36po3plr6uofbq7euzoei54zh/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-python-dateutil-2.8.2-xxydvjzy4b3bbclhveqj6x27xpeje36g/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-nest-asyncio-1.5.5-3zezzdfnphz2nawean346ujin7wvq6xg/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jupyter-core-4.11.1-6ayjk2vovvjcwenmz4ggv77ajkcralcs/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-entrypoints-0.4-qncu2zr255lkshiy2ivbjlosq2ze2rxj/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jupyter-client-7.3.5-lunpy5z42lrwa4akcrab255e6gtoc62c/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pure-eval-0.2.2-hgvrldfqkbawx5e6pamumsqh7gknzpls/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-executing-1.1.0-y6hv5ztljveekcgcnyovquvfy6fgxgzm/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-six-1.16.0-2zcyoyxc4wzuzbo2axtwqmq774ek5suj/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-asttokens-2.0.8-c2gdvkpdempzp7ilblooovft5mghzfvc/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-stack-data-0.5.0-v2pajzhomuus2zmxphlodmb3ygi7jrhe/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pygments-2.13.0-y5h7by47meoneu4suovjl6zkby4qgjaf/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-wcwidth-0.2.5-vepor6n2g5ilew44ppd56y4nmaujvjot/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-prompt-toolkit-3.0.31-7ia7y6lj4siglprxrjocnexq43pqggaq/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pickleshare-0.7.5-y7rqzfukwy2usgdyrzudablwbs47egda/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-ptyprocess-0.7.0-ov34tx2ze7pxswe3e3feeyna74aspz5d/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pexpect-4.8.0-64yawraxtrixn6qq5fg3gil2ohkmnyxh/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-traitlets-5.3.0-rq7s5ips7qlteghd5gjn5mmfqgpipu2a/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-matplotlib-inline-0.1.3-pofyqgsn4g2agtftzvknm2l55xmsratu/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-setuptools-59.4.0-lw5ku6h5eicnlncopldkln4ridjrcwob/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-parso-0.8.3-h3hlt6onv663zdjyfp45eepnis3pdzp5/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jedi-0.18.1-bt676ttyexeb6urhnhs3tovmy2dcay2v/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-decorator-5.1.1-xe27r4nd6fncr5hykcaiqoxvrjy3f3ps/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-backcall-0.2.0-arfastb7tv233g6paivhde4viibyq7d2/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-ipython-8.5.0-4a6j2vvn6qwxn55hzk2rhz46n636hck3/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-debugpy-1.6.3-kyw7sxeslp74hew3y72wwkfqpcjuj7kb/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-terminado-0.15.0-7bclqpaaxtu7osxnoyapqtvqftp7327h/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-send2trash-1.8.0-taus5gn5z6hlvfg7qcdyaqmq3qcl5flr/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-prometheus-client-0.12.0-b543cfgqd5avtbvc3f7lsl2ehfozdc4w/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-tinycss2-1.1.1-ccp5yx7qvnwnujj4s7ati4fv6kxlaj56/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pandocfilters-1.5.0-s6coy5huiopimy2x4ajsdzmg2yhss3ie/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pyrsistent-0.18.1-snd65etsgq764hcjdtmfo6mzifuqa7c6/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-attrs-22.1.0-srjfkoijce7i2at4rsts6syhozabwsoa/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jsonschema-4.16.0-5uclqy3s4x6ms2nenriwcd5ab64uqxuc/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-nbformat-5.1.3-couoi43bbguwmubbkw2weao3ozmscf2h/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-nbclient-0.6.7-oojgxekw5nwszpghzlkhxd6kynd4d27h/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-mistune-2.0.4-e2lgxb5qojk57xl7qscqiktx7xkmzgf4/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-lxml-4.9.1-lyubfrvh7rjxxqzcscccozav23lfqjyl/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jupyterlab-pygments-0.2.2-syih6f3qqkv5flnfzey44ma6u2odkfu4/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-zipp-3.8.1-mifh42prx56ajf5iklyelivgux7ovu2p/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-importlib-metadata-4.12.0-srbpoeuvsc3jsas2ylkhav4jz3phlyhl/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-defusedxml-0.7.1-e676oq76gv7ntepqyxlyl4js3wfbtmmi/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-webencodings-0.5.1-hfnuhs5aqeizspiqmf7nnlewsc65bkah/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-bleach-5.0.1-kat3bj4bsfakb4jxxav5rp34oaanqdvt/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-soupsieve-2.3.2.post1-5c4p6gwwtccwi4ocsxhwki5frsxsurik/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-beautifulsoup4-4.11.1-i7kx65d4vhlawztz6rbjeccopayqg2h5/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-nbconvert-7.0.0-fgxq4toha4iazwhfyuzithzlgcl3vjey/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-markupsafe-2.1.1-66amzezcgjkssy7ssqj73uqgjug5l2y3/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jinja2-3.1.2-ogblkngut6asq4bop5fprv3qfuhj4fog/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-ipython-genutils-0.2.0-abh3xujzt6s3ycxrvxuh2iyu44hkgubr/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-ipykernel-6.16.0-js6r2labegtniighxht4t2y35vbm6tfa/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-argon2-cffi-bindings-21.2.0-m4ygrrcbxwz2owamx7mteztvkrbej5xv/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-argon2-cffi-21.3.0-gs26vp2u3p4q3qvmvcl7oini6novy5cv/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-notebook-6.4.12-e3pl6osm5knvhlrmxoygt24wd6rg5btj/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-websocket-client-1.4.1-5gmmwx75emmdejcpf6ydth6audrozyrn/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-sniffio-1.3.0-z3dxiwo36dh72vl7zpyo4m3xdp7taygz/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-idna-3.4-adfaryoaczgjyrnkmvcmy3wq26vdayf4/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-anyio-3.6.1-vatyumik6kpkfxnmbbbpaykycjuxt2cd/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jupyter-server-1.13.5-wjeijnepbapahk57or6eujjgfrvxzy75/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pytz-2022.2.1-g7e3jrgmlbxvoxqitdbteqzkdxm7hjmz/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-urllib3-1.26.12-zuqhy5lvdowh2krgp4md7g3fghy7ymfn/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-charset-normalizer-2.0.12-dpdwf5pbyvdn2olrzxjdnnikjq2ywpnh/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-certifi-2022.9.14-ng733i6ovywa577qrtae2c2et4wgx3mg/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-requests-2.28.1-f5iulx4ymikjbqdr6tqsrwo5jxix3edt/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-json5-0.9.10-nyn2uzc2tqfbq6tkqrtdedr3hri6xw5u/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-babel-2.10.3-hloghim4tgzt22qciaejohyz7xcrqppt/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-nbclassic-0.3.5-v5qjxr3pebelmdgh2txtdoir45c4iool/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-jupyterlab-server-2.10.3-6ffgbdup5ymjyoextiug2zdfxgrrlgrf/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-filelock-3.5.0-fyoejzpel4yspmbxivwallcmnamjgrwd/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-typing-extensions-4.3.0-u3cfejse2vnwzhiozepvpcgokqys7a4k/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-tqdm-4.64.1-xi66xzv55ytpps3sldw2cqjvqr2mh5w5/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pyyaml-6.0-xt6bta2rqlgkozzmv5qmy4kbm6nheejw/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pybind11-2.6.2-4pvgeryv3fenlan4nfqo3wxdvx4rfgmr/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-protobuf-3.20.1-cucrswilpq4rl6vxtpegdgxtlkz6edl2/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-future-0.18.2-b5b57nvcks3ub3pj3wlup4i3x7vm67ay/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-torch-1.10.1-2cqyvsgcyq2s4q6ggi4hlwzhid5xz6ok/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-pillow-9.2.0-wuypn7qrzhoyahm3ayhwgbpm35qwwefv/lib/python3.9/site-packages:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/py-numpy-1.23.4-64bbtiema7msvubtt4gz2gzifvqbyxum/lib/python3.9/site-packages:/p/lustre1/nexouser/nexo-spack-env-v0.19/.spack-env/view/python:/p/lustre1/nexouser/spack-v0.19/opt/spack/linux-rhel8-broadwell/gcc-10.3.1/root-6.26.06-xvhjetx434h3djveu4pa752oyydpmpx6/lib/root:/p/lustre1/nexouser/nexo-spack-env-v0.19/.spack-env/view/lib/root'.split(':')
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from DnnEventTagger import Datasets

from networks.resnet_example import resnet18
# from utils.data_loaders import DenseDataset
# from utils.data_loaders import DatasetFromSparseMatrix,NoisedDatasetFromSparseMatrix
import traceback
import yaml 

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
list_of_gpus = range(torch.cuda.device_count())
device_ids = range(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = str(list_of_gpus).strip('[]').replace(' ', '')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs = 200

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = lr/np.exp(epoch/10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(trainloader, epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, '/', len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader), 100.*correct/total

def test(testloader, epoch, saveall=False):
    
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    score = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            softmax = nn.Softmax(dim=0)
            for m in range(outputs.size(0)):
                score.append([softmax(outputs[m])[1].item(), targets[m].item()])
                # score.append([outputs[m][1].item(), targets[m].item()])
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    is_better = False
    acc = 100.*correct/total
    
    # If we want to save all training records
    if saveall:
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(filename_prefix):
            os.mkdir(filename_prefix)
        if not os.path.isdir(filename_prefix):
            os.mkdir(filename_prefix)
        torch.save(state, './output' + shortname_prefix[:-1] + '/checkpoints' + shortname_prefix + ('ckpt_%d.t7' % epoch))
        torch.save(state, './output' + shortname_prefix[:-1] + '/checkpoints' + shortname_prefix + 'ckpt.t7')
        best_acc = acc
    # Otherwise only save the best one
    elif acc > best_acc:
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(filename_prefix):
            os.mkdir(filename_prefix)
        if not os.path.isdir(filename_prefix):
            os.mkdir(filename_prefix)
        torch.save(state, './output' + shortname_prefix[:-1] + '/checkpoints' + shortname_prefix + ('ckpt_%d.t7' % epoch))
        torch.save(state, './output' + shortname_prefix[:-1] + '/checkpoints' + shortname_prefix + 'ckpt.t7')
        best_acc = acc
        
    return test_loss/len(testloader), 100.*correct/total, score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', '-f', type=str, default="noise_test.yml", help="specify yaml config")
    # parser.add_argument('--noise_amp', '-n', type=float, default=100, help="Set noise level in electrons")
    parser.add_argument('--prefix', '-p', type=str, default='lmdb_mislabel_', help="Set filename prefix")
    parser.add_argument('--submit', '-s', action='store_true',help='submit job')
    # parser.add_argument('--restore_quiet', '-q', action='store_true',help='Add 25 unit gaussian noise to quiet channels')
    # parser.add_argument('--reseed_quiet', '-Q', action='store_true',help='1_\{123\} for restoring quiet channels (reseed restoration of RMS noise each epoch)')
    # parser.add_argument('--reseed_noise', '-N', action='store_true',help='X_\{123\} for excess noise (reseed excess noise each epoch)')
    parser.add_argument('--run', '-R', action='store_true',help='run nEXOClassifier')
    parser.add_argument('--mislabeled_gammas_as_electrons', '-m', type=float, default=0, help='Percent of physics data to be gammas')

    args = parser.parse_args()

    config = yaml_load(args.config)
    filename_prefix = config['save_dir'] + args.prefix + ('%g'%(args.mislabeled_gammas_as_electrons))
    shortname_prefix = '/' + args.prefix + ('%g'%(args.mislabeled_gammas_as_electrons)) + '_'

    if not os.path.isdir(filename_prefix):
        os.mkdir(filename_prefix)
    if not os.path.isdir(filename_prefix + '/checkpoints'):
        os.mkdir(filename_prefix + '/checkpoints')

    if args.run:

        # parameters
        data_name = config['data']['name']
        in_name1 = config['data']['in_name1'].split()
        in_name2 = config['data']['in_name2'].split()
        in_name3 = config['data']['in_name3'].split()
        # in_name = '/p/lustre2/nexouser/scotswas/DNN_study/output/%ie/images'%(100)
        # h5file = config['data']['h5name']
        # fcsv = config['data']['csv']
        input_shape = [int(i) for i in config['data']['input_shape']]
        lr = config['fit']['compile']['initial_lr']
        batch_size = config['fit']['batch_size']
        epochs = config['fit']['epochs']
        
        gammas_in_physics = args.mislabeled_gammas_as_electrons / 100


        # Data
        print('==> Preparing data..')
        
        nEXODatasetHolder = []
        nEXODataset1_len = 0
        nEXODataset2_len = 0
        nEXODataset3_len = 0
        # print(in_name1,in_name2,in_name3)
        for in_name in in_name1:
            nEXODatasetHolder += [Datasets.pyxis_discriminator_dataset(in_name)]
            nEXODataset1_len += len(nEXODatasetHolder[-1])
        for in_name in in_name2:
            nEXODatasetHolder += [Datasets.pyxis_discriminator_dataset(in_name)]
            nEXODataset2_len += len(nEXODatasetHolder[-1])
        for in_name in in_name3:
            nEXODatasetHolder += [Datasets.pyxis_discriminator_dataset_of_betas(in_name)]
            nEXODataset3_len += len(nEXODatasetHolder[-1])

        nEXODataset = torch.utils.data.ConcatDataset(nEXODatasetHolder)

        shuffle_dataset = True
        random_seed = 40
        indices1 = np.arange(nEXODataset1_len)
        indices2 = np.arange(nEXODataset2_len) + nEXODataset1_len
        indices3 = np.arange(nEXODataset3_len) + nEXODataset1_len + nEXODataset2_len
        # print(indices1,indices2,indices3)
        validation_split = .2
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices1)
            np.random.shuffle(indices2)
            np.random.shuffle(indices3)
        val_indices = np.concatenate((indices1[:round(len(indices1)*validation_split)],indices2[:round(len(indices2)*validation_split)]))
        train_indices = np.concatenate((indices1[round(len(indices1)*validation_split):],indices2[round(len(indices2)*validation_split):len(indices2)-round(len(indices2)*gammas_in_physics)],indices3[:round(len(indices2)*gammas_in_physics)]))
        train_indices = np.array(train_indices)

        print('random_seed:',random_seed)#,'seed1:', seed1,'seed2:', seed2)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
        validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=validation_sampler, num_workers=0)

        start_epoch = 0

        print('==> Building model..')
        # net = preact_resnet.PreActResNet18(num_channels=args.channels)
        net = resnet18(input_channels=input_shape[2])
        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        # We use SGD
        # optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), 
                                    weight_decay=1e-4, eps=1e-08, amsgrad=False)

        net = net.to(device)
        
        if torch.cuda.device_count() > 1:
            print("Let's use ", torch.cuda.device_count(), " GPUs!")
            net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
            
        if args.resume and os.path.exists(filename_prefix + '/checkpoints' + shortname_prefix + 'ckpt.t7'):
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir(filename_prefix), 'Error: no checkpoint directory found!'
            if device == 'cuda':
                checkpoint = torch.load(filename_prefix + '/checkpoints' + shortname_prefix + 'ckpt.t7' )
            else:
                checkpoint = torch.load(filename_prefix + '/checkpoints' + shortname_prefix + 'ckpt.t7', map_location=torch.device('cpu') )
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            
        # Numpy arrays for loss and accuracy, if resume from check point then read the previous results
        if args.resume and os.path.exists(filename_prefix + shortname_prefix + 'loss_acc.npy'):
            arrays_resumed = np.load(filename_prefix + shortname_prefix + 'loss_acc.npy', allow_pickle=True)
            y_train_loss = arrays_resumed[0]
            y_train_acc  = arrays_resumed[1]
            y_valid_loss = arrays_resumed[2]
            y_valid_acc  = arrays_resumed[3]
            test_score   = arrays_resumed[4].tolist()
        else:
            y_train_loss = np.array([])
            y_train_acc  = np.array([])
            y_valid_loss = np.array([])
            y_valid_acc  = np.array([])
            test_score   = []
        
        for epoch in range(start_epoch, start_epoch + epochs):

            # if args.reseed_noise:
            #     sq1 = np.random.SeedSequence()
            #     nEXODataset.seed_list1 = sq1.generate_state(dataset_size)
            # if args.restore_quiet and args.reseed_quiet:
            #     sq2 = np.random.SeedSequence()
            #     nEXODataset.seed_list2 = sq2.generate_state(dataset_size)

            # Set the learning rate
            adjust_learning_rate(optimizer, epoch, lr)
            iterout = "\nEpoch [%d]: "%(epoch)
            
            for param_group in optimizer.param_groups:
                iterout += "lr=%.3e"%(param_group['lr'])
                print(iterout)
                try:
                    train_ave_loss, train_ave_acc = train(train_loader, epoch)
                except Exception as e:
                    print("Error in training routine!")
                    print(e.message)
                    print(e.__class__.__name__)
                    traceback.print_exc(e)
                    break
                print("Train[%d]: Result* Loss %.3f\t Accuracy: %.3f"%(epoch, train_ave_loss, train_ave_acc))
                y_train_loss = np.append(y_train_loss, train_ave_loss)
                y_train_acc = np.append(y_train_acc, train_ave_acc)

                # Evaluate on validationset
                try:
                    valid_loss, prec1, score= test(validation_loader, epoch, True)
                except Exception as e:
                    print("Error in validation routine!")
                    print(e.message)
                    print(e.__class__.__name__)
                    traceback.print_exc(e)
                    break
                    
                print("Test[%d]: Result* Loss %.3f\t Precision: %.3f"%(epoch, valid_loss, prec1))
                
                test_score.append(score)
                y_valid_loss = np.append(y_valid_loss, valid_loss)
                y_valid_acc = np.append(y_valid_acc, prec1)
                
                np.save(filename_prefix + shortname_prefix + 'loss_acc.npy', np.array([y_train_loss, y_train_acc, y_valid_loss, y_valid_acc, test_score], dtype=object))
    
    else:
        job = filename_prefix+shortname_prefix[:-1]+'.sh'
        jobName = shortname_prefix[1:-1]
        systemOut = "-o %s.sout -e %s.serr" % (filename_prefix + shortname_prefix[:-1], filename_prefix + shortname_prefix[:-1])
        cmd = "%s -J %s %s %s" % ('/usr/global/tools/flux_wrappers/bin/sbatch -t 1-00:00:00', jobName, systemOut, job)
        # cmd = "%s -J %s %s %s" % ('/usr/bin/sbatch -t 3-00:00:00', jobName, systemOut, job)
        
        print(filename_prefix+shortname_prefix[:-1]+'.sh')

        with open(filename_prefix+shortname_prefix[:-1]+'.sh','w') as file:
            file.write("""#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -p pdebug
#SBATCH -J %s

module load rocm/5.2.3
source /p/vast1/nexo/tioga_software/tioga-torch/bin/activate
cd /p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL
source setup.sh
cd scripts
python nEXOClassifier_lmdb_mixedLabels.py %s--config %s --mislabeled_gammas_as_electrons %s -p %s -R > %s.out 2> %s.err""" % (jobName,'-r '*args.resume,args.config,args.mislabeled_gammas_as_electrons,args.prefix,filename_prefix + shortname_prefix[:-1], filename_prefix + shortname_prefix[:-1]))

        if args.submit:
            print(cmd)
            os.system(cmd)
        else:
            print(cmd)