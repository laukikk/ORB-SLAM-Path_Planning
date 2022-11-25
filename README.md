## Installation

Install the basic Python Dependencies
```
pip install -r requirements.txt
```

Other Installations
```
sudo apt-get install libqglviewer-dev-qt5
```

### g2oPy

Clone the repository
```
git clone https://github.com/uoip/g2opy.git
```

Replace the following code with the code after that at `g2opy/python/core/eigen_types.h`
```
.def("x", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::x)
.def("y", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::y)
.def("z", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::z)
.def("w", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::w)
```
with
```
.def("x", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::x)
.def("y", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::y)
.def("z", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::z)
.def("w", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::w)
```

### Pangolinn

Do [this](https://github.com/uoip/pangolin/issues/5#issuecomment-909434057)


## TO-DO:
- Add docs for g2oPy
- Add docs for Pangolin