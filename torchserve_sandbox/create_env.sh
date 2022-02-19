# Note: There is no DLAMI Conda environment for this framework version
#       Framework will be installed/updated inside a Python environment

# Update OS packages
sudo apt-get update -y

###############################################################################################################
# Before installing or updating aws-neuron-dkms:
# - Stop any existing Neuron runtime 1.0 daemon (neuron-rtd) by calling: 'sudo systemctl stop neuron-rtd'
###############################################################################################################

################################################################################################################
# To install or update to Neuron versions 1.16.0 and newer from previous releases:
# - DO NOT skip 'aws-neuron-dkms' install or upgrade step, you MUST install or upgrade to latest Neuron driver
################################################################################################################

# Install OS headers
sudo apt-get install linux-headers-$(uname -r) -y

# Install Neuron Driver
sudo apt-get install aws-neuron-dkms -y

####################################################################################
# Warning: If Linux kernel is updated as a result of OS package update
#          Neuron driver (aws-neuron-dkms) should be re-installed after reboot
####################################################################################

# Install Neuron Tools
sudo apt-get install aws-neuron-tools -y

export PATH=/opt/aws/neuron/bin:$PATH

# Activate PyTorch
source activate aws_neuron_pytorch_p36

# Set Pip repository  to point to the Neuron repository
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

#Install Neuron PyTorch
pip install torch-neuron neuron-cc[tensorflow] torchvision