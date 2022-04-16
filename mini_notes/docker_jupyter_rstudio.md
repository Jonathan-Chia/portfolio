Created by Chia, Jonathan, last modified on Apr 16, 2022

# Introduction

**With 2 lines of code you can:**

*   **Set up Jupyter Notebook preloaded with tensorflow, numpy, pandas, sklearn, etc.**

*   **Set up R Studio preloaded with tidyverse packages**

* [Steps](#steps)
  * [0. Install Requirements](#req)
  * [1. Install Docker](#install)
  * [2. Pull Docker Image](#pull)
  * [3. Run Docker Image](#run)
  * [4. Start the Container](#start_container)
  * [5. Sign On](#sign_on)
* [Next Steps](#next_steps)
  * [1. Environment Setup](#environments)
  * [2. Install Additional Packages](#additional_packages)
  * [3. Install Additional Packages through Docker File](#additional_packages_docker)
* [Additional Resources](#additional_resources)

# Steps <a name="steps"></a>
======

## 0. Install Requirements <a name='req'></a>

See [https://docs.docker.com/desktop/](https://docs.docker.com/desktop/)

Click on Mac, Windows, or Linux to see instructions for installing system requirements.

Install all requirements.

## 1. Install Docker <a name="install"></a>
-----------------

Download Docker Desktop:

[https://www.docker.com/get-started](https://www.docker.com/get-started)

  

You will see three different tabs:

Containers - virtualized run-time environment where users can isolate applications from the underlying system. Basically **where you run the template**.

Images - an immutable (unchangeable) file that contains the source code, libraries, dependencies, tools, and other files needed for an application to run. Basically a **template.**

Volumes - this is where the data is stored.

  

[https://phoenixnap.com/kb/docker-image-vs-container](https://phoenixnap.com/kb/docker-image-vs-container)

[https://docs.docker.com/storage/volumes/](https://docs.docker.com/storage/volumes/)

  

Note: If Docker is not working, make sure 'virtualization' is enabled in your BIOS. 

## 2. Pull Docker Image <a name="pull"></a>
---------------------

We will now pull pre-built images from these sources:
* [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/)
* [Rocker Project](https://www.rocker-project.org/images/)

Open up your command line (in windows you can open up windows powershell).

Run the below code:

R Studio:

```linux
docker pull rocker/tidyverse
```

Jupyter Lab:

```linux
docker pull jupyter/tensorflow-notebook
```

You should now see _rocker/tidyverse_ or _jupyter/tensorflow-notebook_ in your _Images_:

![](attachments/95650216/95650222.png)

Note: you can choose to install a different image if you would like. 
For example, if you don't need tensorflow you can install [jupyter/scipy-notebook](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-scipy-notebook).

## 3. Run Docker Image <a name="run"></a>
--------------------

Now we need to run the image to create a container:

R Studio:

```linux
docker run -p 8787:8787 -v r_volume:/app -e PASSWORD=rstudio --name rstudio rocker/tidyverse
```

-p tells it to run in the browser

-v names the volume

--name names the container

Jupyter Lab:

```linux
docker run -p 8888:8888 --name jupyter -v jupyter_volume:/app -e JUPYTER_ENABLE_LAB=yes -it jupyter/tensorflow-notebook
```

-v names the volume and sets it up

-it makes it interactive

Now the data will be stored in _Volumes_:
-----------------------------------------

![](attachments/95650216/95650221.png)
--------------------------------------

## 4. Start the Container <a name="start_container"></a>
-----------------------

Click the _Play_ button:
------------------------

![](attachments/95650216/95650220.png)
--------------------------------------

And then click _Open in Browser:_
---------------------------------

![](attachments/95650216/95650219.png)
----------------------------------------

## 5. Sign On <a name="sign_on"></a>
-----------

Jupyter: 

If Jupyter Lab asks for a token, look at the command line and you'll see it in one of the outputs.

  

R Studio:

Username: rstudio

Password: rstudio

  

If everything was set up correctly, you should see these screens:
-----------------------------------------------------------------

Jupyter Lab:

![](attachments/95650216/95650218.png)

  

R Studio:

![](attachments/95650216/95650217.png)

# Next Steps <a name="next_steps"></a>
===========

## 1. Set up Environments: <a name="environments"></a>
--------------------

Set up your environments.

  

## 2. Installing Additional Packages: <a name="additional_packages"></a>
-------------------------------

Jupyter Lab:

Use pip install in the jupyter lab terminal

  

R Studio:

Use install.packages()

  

I believe the packages are saved in the volume.

## 3. Additional Package Installation through Docker Files: <a name="additional_packages_docker"></a>
-----------------------------------------------------

If you want the container to start with certain packages you can use a Docker File to do that.

  

Don't have an article for this yet, but see here:

[https://davetang.org/muse/2021/04/24/running-rstudio-server-with-docker/](https://davetang.org/muse/2021/04/24/running-rstudio-server-with-docker/)

[https://towardsdatascience.com/how-to-run-jupyter-notebook-on-docker-7c9748ed209f](https://towardsdatascience.com/how-to-run-jupyter-notebook-on-docker-7c9748ed209f)

  

  

  

  

  

# Additional Resources <a name="additional_resources"></a>

See this link:

[https://jupyter-docker-stacks.readthedocs.io/en/latest/](https://jupyter-docker-stacks.readthedocs.io/en/latest/)

  

Docker Tutorial for R:

[https://jsta.github.io/r-docker-tutorial/02-Launching-Docker.html](https://jsta.github.io/r-docker-tutorial/02-Launching-Docker.html)


---
Document generated by Confluence on Apr 09, 2022 16:54

[Atlassian](http://www.atlassian.com/)
