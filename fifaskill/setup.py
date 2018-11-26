import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fifaskill",
    version="0.0.1",
    author="Vinay Ramesh",
    author_email="vrr2112@columbia.edu",
    description="A Package for Fifa Team Skill Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinoo999/trueskill_augmented",
    packages=['fifaskill', 'fifaskill.data_processing',
              'fifaskill.models'],
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
)
