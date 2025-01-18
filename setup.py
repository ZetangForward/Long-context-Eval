# # zecheng_note: longtexteval.eval --task counting_star --output_dir=/opt/data/private/sora/lab/longtexteval/output/2024-12-08 21-18_results/L-CiteEval-Hardness_gov_report
# # zecheng_note: longtexteval.eval --generation --eval --save_input
# import setuptools
# import os
import setuptools


# This is to make sure that the package supports editable installs
setuptools.setup()

# with open("./requirements.txt", 'r', encoding='utf-8') as f:
#     requirements = f.read().strip().splitlines()
# with open("./README.md", "r") as f:
#     long_description = f.read()
# import setuptools
# import os

# print("当前工作目录:", os.getcwd())
# with open("./requirements.txt", 'r', encoding='utf-8') as f:
#     requirements = f.read().strip().splitlines()
# with open("./README.md", "r") as f:
#     long_description = f.read()

# setuptools.setup(
#     name="longtexteval",
#     version="0.1",
#     author="LongTextEval Team",
#     description="A useful longtexteval demo",
#     long_description=long_description,
#     packages=setuptools.find_packages(),
#     python_requires=">=3.10",
#     install_requires=requirements,
#     entry_points={
#         'console_scripts': [
#             'lte.eval = longtexteval.main:main',
#             'lte = longtexteval.main:main'  # 假设main.py在longtexteval包目录下，且入口函数是main
#         ]
#     }
# )