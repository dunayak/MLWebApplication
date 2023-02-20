# sudo apt install git 
sudo firewall-cmd --zone=public --add-port=80/tcp --permanent
echo "Firewall rule added"
sudo firewall-cmd --reload
echo "Firewall reloaded"
sudo apt-get update 
echo "Package Manager updated"
sudo apt-get install python3
echo "python3 installed"
sudo apt-get install python3 pip
echo "python3 pip installed"
sudo pip3 install flask==0.12.2
echo "Flask installed"
sudo pip3 install numpy
echo "numpy installed"
sudo pip3 install -U scikit-learn scipy matplotlib
echo "scikit-learn installed"
sudo pip3 install pandas
echo "pandas installed"
sudo pip3 install flask_caching
echo "flask caching installed"
git clone https://github.com/dunayak/MLWebApplication
echo "Git Repository cloned"
