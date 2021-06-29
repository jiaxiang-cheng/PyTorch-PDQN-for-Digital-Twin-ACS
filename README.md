# Conveyor System Control with Deep Q-Networks
PyTorch implementation of optimized range-inspection control (RIC) or look-ahead control for digital twin automated conveyor system (DT-ACS) with production station as agent, or conceyor-serviced production system (CSPS), using Deep Q-Networks (DQN) and Profit-Sharing (PS), or PDQN for short.  
_Author: Wang, Tian and Cheng, Jiaxiang and Yang, Yi and Esposito, Christian and Snoussi, Hichem and Tao, Fei*_

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" /> 

Official repository for our publications:   
- [Adaptive Optimization Method in Digital Twin Conveyor Systems via Range-Inspection Control (2020)](https://ieeexplore.ieee.org/abstract/document/9303438)   
- [Optimal Look-ahead Control of CSPS System by Deep Q-Network and Profit Sharing (2018)](https://ieeexplore.ieee.org/abstract/document/8623593)

## Environment

The environment or the requirements for running the project:

- python==3.8.10  
- numpy==1.20.2   
- matplotlib==3.3.4
- pytorch==1.8.1

## Usage

We have made improvements on the whole coding with implementation of Python and PyTorch. So the simulation with all the 
methodologies used in our work can be executed easily with the following actions:

### (1) Simulation with PDQN (_DQN + PS_)
```
python main.py
python main.py --model 1 --ps 1
```
The default methodology used is PDQN without any argument parser highlighted.

### (2) Simulation with DQN
```
python main.py --model 1 --ps 2
```
### (3) Simulation with QL (_with PS_)
```
python main.py --model 2 --ps 1
```
### (4) Simulation with QL
```
python main.py --model 2 --ps 2
```
We indicated the total rewards as the criterion for comparison among performance of different methodologies.

## Citation
```
@article{wang2020adaptive,
  title={Adaptive Optimization Method in Digital Twin Conveyor Systems via Range-Inspection Control},
  author={Wang, Tian and Cheng, Jiaxiang and Yang, Yi and Esposito, Christian and Snoussi, Hichem and Tao, Fei},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2020},
  publisher={IEEE}
}
```
## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Credit

Our work has been built on the great work from Tang, Hao team from Hefei University of Technology, 
as they have done excellent work on modeling the look-ahead control problem for conveyor-serviced production system with semi-Markov process.
You may refer to one of their work: Tang, H., Xu, L., Sun, J., Chen, Y., & Zhou, L. (2015). Modeling and optimization control of a demand-driven, conveyor-serviced production station. European Journal of Operational Research, 243(3), 839-851.
