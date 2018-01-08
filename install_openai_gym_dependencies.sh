apt-get install -y python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev\
  xvfb libav-tools xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig
pip install --upgrade 'gym[all]'
# Requirement for certain gyms
pip3 install box2d box2d-kengz
