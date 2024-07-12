from distutils.core import setup
setup(
  name='exetest',         # How you named your package folder (MyLib)
  packages=['exetest'],   # Chose the same as "name"
  version='0.9.8',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description='A pytest-based test framework for black-box approach to testing executables',   # Give a short description about your library
  author='Guillaume227',                   # Type in your name
  author_email='guillaume227@gmail.com',      # Type in your E-Mail
  url='https://github.com/Guillaume227/exetest',   # Provide either the link to your github or to your website
  download_url='https://github.com/Guillaume227/exetest/archive/v0.1-alpha.tar.gz',
  keywords=['test', 'pytest', 'nosetest'],  # Keywords that define your package best
  install_requires=[            # I get to this in a second
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify the python versions that you want to support
  ],
  python_requires='>=3.6'
)
