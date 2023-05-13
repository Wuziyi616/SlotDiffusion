from setuptools import setup, find_packages

requirements = [
    'numpy', 'pyyaml', 'six', 'tqdm', 'opencv-python', 'matplotlib', 'open3d',
    'pycocotools', 'pandas', 'moviepy', 'imageio', 'scikit-image', 'wandb',
    'torch', 'torchvision', 'torchmetrics', 'lpips', 'einops', 'transformers',
    'pytorch-lightning', 'nerv'
]

title = "SlotDiffusion: Object-Centric Learning with Diffusion Models"
setup(
    name="slotdiffusion",
    version='0.1.0',
    description=title,
    long_description=title,
    author="Ziyi Wu",
    author_email="ziyiwu@cs.toronto.edu",
    license="MIT",
    url="https://slotdiffusion.github.io/",
    keywords="object-centric diffusion model",
    packages=find_packages(),
    install_requires=requirements,
)
