To execute the CUDA implementation, please run the 'make' file. 

Since this implementation makes use of Nvidia's dynamic parallelism functionality, the file is compiled differently... As such, it appears that dynamic parallelism is a known issue when executed on an Ubuntu system( this is noted on numerous nvidia's forums and websites and we cannot take responsibility for this). However, this implementation was implemented on a Windows computer with Windows Powershell and works flawlessly... This issue was only discovered on the day of the submission and I unfortunately do not have access to a functioning Ubuntu device with CUDA ... As a result, should you wish to execute the implementation, the implementation works fine on the cluster. 

To clean the executable, please execute 'make clean'
