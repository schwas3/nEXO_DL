#%% 
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

#%%
mislabeled_gammas_as_electrons = 20
prefix = 'lmdb_mislabel_'
config = yaml_load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/config/lmdb_mislabels.yml')
filename_prefix = config['save_dir'] + prefix + ('%g'%(mislabeled_gammas_as_electrons))
shortname_prefix = '/' + prefix + ('%g'%(mislabeled_gammas_as_electrons)) + '_'

# parameters
data_name = config['data']['name']
in_name1 = config['data']['in_name1'].split()
in_name2 = config['data']['in_name2'].split()
in_name3 = config['data']['in_name3'].split()
# in_name1 = ['/p/lustre2/nexouser/scotswas/DNN_study/output/test4/images','/p/lustre2/nexouser/scotswas/DNN_study/output/test4/images']
# in_name2 = ['/p/lustre2/nexouser/scotswas/DNN_study/output/test4/images','/p/lustre2/nexouser/scotswas/DNN_study/output/test4/images']
# in_name3 = ['/p/lustre2/nexouser/scotswas/DNN_study/output/test4/images']
# in_name = '/p/lustre2/nexouser/scotswas/DNN_study/output/%ie/images'%(100)
# h5file = config['data']['h5name']
# fcsv = config['data']['csv']
input_shape = [int(i) for i in config['data']['input_shape']]
lr = config['fit']['compile']['initial_lr']
batch_size = config['fit']['batch_size']
epochs = config['fit']['epochs']

gammas_in_physics = mislabeled_gammas_as_electrons / 100


# Data
print('==> Preparing data..')
        
# nEXODataset1 = []
# nEXODataset2 = []
# nEXODataset3 = []
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
    if in_name2.index(in_name) == len(in_name2) - 1: nEXODatasetHolder[-1].len = round(.8 * nEXODataset2_len)
    nEXODataset2_len += len(nEXODatasetHolder[-1])
for in_name in in_name3:
    nEXODatasetHolder += [Datasets.pyxis_discriminator_dataset_of_betas(in_name)]
    nEXODataset3_len += len(nEXODatasetHolder[-1])

nEXODataset = torch.utils.data.ConcatDataset(nEXODatasetHolder)

shuffle_dataset = True
#%%
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
print(val_indices.shape)
print(train_indices.shape)
print(len(indices1[:round(len(indices1)*validation_split)]))
print(len(indices2[:round(len(indices2)*validation_split)]))
print(len(indices1[round(len(indices1)*validation_split):]))
print(len(indices2[round(len(indices1)*validation_split):len(indices2)-round(len(indices2)*gammas_in_physics)]))
print(len(indices3[:round(len(indices2)*gammas_in_physics)]))
print(round(len(indices2)*gammas_in_physics))
# print(train_indices)
# print(val_indices)
np.random.shuffle(val_indices)
np.random.shuffle(train_indices)
# print(len(train_indices))
# print(len(val_indices))

print(len(nEXODataset))
print(len(val_indices))
print(len(train_indices))
print(len(train_indices)+len(val_indices))

print('random_seed:',random_seed)#,'seed1:', seed1,'seed2:', seed2)

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=validation_sampler, num_workers=0)

start_epoch = 0

# print('==> Building model..')
# net = preact_resnet.PreActResNet18(num_channels=args.channels)
# net = resnet18(input_channels=input_shape[2])

# %%
