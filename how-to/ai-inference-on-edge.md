
# AI Inference on the Edge with TensorFlow Lite

This blog post dives into the world of AI on the edge, and how to deploy TensorFlow Lite models on edge devices. We’ll explore the challenges of managing dependencies and updates for these models, and how containerisation with Ubuntu Core and Snapcraft can streamline the process.

Let’s start by defining what TensorFlow and its Lite variant are.

TensorFlow and its sibling TensorFlow Lite
TensorFlow is a machine learning platform that implements the current best practices. It provides tools for creating ML models, running them, monitoring and improving them. TensorFlow aims to assist beginners and professionals deploying to production environments on desktop, mobile, web and cloud.

TensorFlow Lite is a library meant for running ML models on the edge or on platforms where resource constraints are greater, for example microcontrollers, embedded systems, mobile phones and so on. TF Lite is ideal when the only thing you need to do is to run an ML model. The TensorFlow Lite runtime is a fraction of the size of the full TensorFlow package and includes the bare minimum features to run inference while saving disk space. TF Lite can also optimise existing TensorFlow models by using quantization methods. This reduces the required computing resources, while only incurring a negligible loss in accuracy.

There are two main challenges in bringing TF Lite inference to production: dependency and update management.

Dependency issues
TensorFlow is a large machine learning framework with hundreds of dependencies. In comparison, the TensorFlow Lite runtime has a smaller set of dependencies. Let’s take a look at a few of the dependencies for tflite-runtime and tflite-support, the two main libraries required in many deployments: 

tflite-runtime: a simplified library for running machine learning models on mobile and embedded devices, without having to include all TensorFlow packages. 
wheel ~ 2MB
tflite-support: a toolkit that helps users develop and deploy TFLite models onto mobile devices. Even though tflite-runtime should be enough for most use cases, tflite-support adds extra functionality to customise how the model is run. This is especially useful when using a hardware accelerator.
wheel ~ 270MB
Being a Python framework, Tensorflow’s most important dependency is obviously the Python runtime. Both tflite-runtime and tflite-support depend on Python <=3.11, while tflite-support on ARM64 works on Python <=3.9 only. Ubuntu 24.04 which is the current Ubuntu release with the longest support ships Python 3.12.

Another key dependency is NumPy. The tflite-runtime library requires NumPy 1.x, while the latest version is v2.x. If you naively install tflite-runtime, it will install the latest NumPy resulting in the following runtime error:

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.0 as it may crash. To support both 1.x and 2.x versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
If you want to work with image files, you also need the Python image library called Pillow.

One of the powerful features of TF Lite is its ability to use Tensor Processing Units (TPUs)  for offloading of the computations. TPUs are silicons designed for efficient neural network calculations. Use of TPUs from TF Lite on Linux requires the drivers and supporting libraries. For example, to use a Coral EdgeTPU USB Accelerator, you need Edge TPU runtime and the PyCoral Python library:

Edge TPU runtime provides the core programming interface for the Edge TPU. There are two Debian packages for the drivers: libedgetpu1-std (slower clock rate, lower power, less heat), and libedgetpu1-max (higher clock rate, higher TPU temperature).
PyCoral is a Python library built on top of the TensorFlow Lite library to speed up your development and provide extra functionality for the Edge TPU. PyCoral currently supports only Python 3.6 through 3.9.
Bundling the correct version of all these drivers and libraries into your software can be a daunting task. Especially if you have multiple software packages running on the same system, requiring different versions of the same library. For the most TensorFlow examples we need Python 3.8 for full compatibility with all the dependencies, but some other system software such as the Raspberry Pi 5 fan controller, might require Python 3.12.

Update management challenges
Machine learning models are created based on training datasets. In many applications, the machine learning models need to evolve to maintain their prediction accuracy based on current observations and trends. Continuous machine learning often relies on feedback loops and incremental training pipelines; this isn’t easily possible on resource-constrained devices on the edge. That’s why in many scenarios, it makes sense to develop a workflow to remotely update the machine learning models, similar to every other software components and dependencies.

Another limiting factor is the device deployment location. Edge devices that perform ML inference could be located in remote areas, far apart, or inside factories or data centres with strict access control. This makes physical updates of these devices difficult and expensive.

Usually, the initial developments take over-the-air updates and remote access into account. However, the delivery mechanisms aren’t atomic and transactional. Uploading a new model or new software to the edge is always risky. There could be connection issues during the update, or the startup may fail due to an unknown bug that surfaced only in the field. Updates that result in failures can be expensive to resolve, if not catastrophic.

An industry standard for dependency and update management
Dependency and update management are widespread challenges in this industry. There are many ways to tackle the various issues. Here, we leverage Ubuntu Core and the snap ecosystem, designed with security and remote management in mind. Ubuntu Core is fully containerised, making it possible to reliably update key building blocks of the operating system, ranging from the kernel itself to system packages. The native packaging format on Ubuntu Core is snap. Snaps are not only used for applications but also the operating system components. This offers an end-to-end and consistent update mechanism for the entire software stack. Let’s continue by creating a snap for TF Lite application.

Snapping a TF Lite application
A snap is a software package that creates a sandboxed environment for the software to run in. Inside this sandbox, only the required dependencies are available for the software to run correctly. It also limits access to the host operating system, with rules to allow certain capabilities such as talking to USB hardware or displaying a GUI on the screen. This solves dependency issues, while drastically improving security.

Let’s package a TensorFlow application which classifies an input image. We assume you already know the basics of building snaps, as explained in the Create a new snap tutorial.

Starting with a base
Our snap uses core24 for its base. A base is the file system and basic set of utilities that are available inside the snap. Core24 inherits this from Ubuntu 24.04 LTS, thus providing up to 12 years of support and security updates.

base: core24
Sorting out Python
The base bundles Python 3.12 which is incompatible with the application. Instead, we are interested in Python 3.8 which is compatible with our, and many of the existing upstream examples. We use the deadsnakes PPA to include the Python 3.8 interpreter inside the snap. The deadsnakes PPA provides older and newer versions of Python for current Ubuntu releases.

package-repositories:
  - type: apt
    ppa: deadsnakes/ppa
Assembling a snap consists of different parts. These can be seen as the steps to follow to put the final snap together. Our first step is to install the Python 3.8 Debian package.

parts:
  python38:
    plugin: nil
    stage-packages:
      - python3.8-full
In another part we use the snapcraft python plugin to install our Python dependencies. We tell it to run after the part that installed Python 3.8, and then specify Python 3.8 as the interpreter the plugin should use. Lastly, we install the packages for the latest Pillow and tflite-runtime, as well as a specific version of NumPy.

python-dependencies:
    after: [python38]
    plugin: python
    build-environment:
      - PARTS_PYTHON_INTERPRETER: python3.8
    python-packages:
      - numpy<2
      - pillow
      - tflite-runtime
Adding your application code
Our business logic, in this case a Python script running the ML workload, goes into its own part. This script is based on the TensorFlow Lite Python image classification demo. It takes an image file as input, runs it through the ML model, which recognises objects in the image, and prints a list of labels and their certainty out to the terminal.

scripts:
    plugin: dump
    source: .
    override-build: |
      cp label_image_lite.py $CRAFT_PART_INSTALL/
Downloading and including the model and labels also get their own parts.

  model:
    plugin: dump
    source: <url>/model.tgz
    source-type: tar
    override-build: |
      cp model.tflite $CRAFT_PART_INSTALL/
  labels:
    ...
You might be wondering why we use so many different parts that do so little, while everything can be done in a single part. We specifically do this to improve build caching and delta updates. We will discuss this in the next section.

In the apps section we define which command is executed when the snap is called. In our case the command is the Python interpreter with the script as its only argument. The plug defines that this application needs access to the user’s home directory.

apps:
  tf-label-image:
    plugs:
      - home
    command: bin/python3 $SNAP/label_image_lite.py
This snap only exposes an app that needs to be interactively called. It is also possible to define a service app. If it is defined as a daemon, the app runs constantly in the background. It can also be automatically started after installation and at boot. This can be useful on an edge device that needs to perform a persistent task, without being interacted with by a user. To change an app to become a service, one needs to update the code to continuously process a stream, like a video feed, and add daemon: simple to the app definition.

apps:
  tf-label-image:
    plugs:
      - home
    command: bin/python3 $SNAP/daemon_script.py
    daemon: simple
Running the snap
We build our snap using snapcraft -v and then install it using snap install --dangerous ./tf-label-image_*.snap. After it’s installed the example can be run. If no input image is provided, it will use an included photo of Grace Hopper. The script should print out a list of labels.


Image source: Wikimedia Commons
$ tf-label-image
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
0.919721: 653:military uniform
0.017762: 907:Windsor tie
0.007507: 668:mortarboard
0.005419: 466:bulletproof vest
0.003828: 458:bow tie, bow-tie, bowtie
time: 28.502ms
You can also provide an image to be labelled.


Image source: Wikimedia Commons, Adrian Pingstone
$ tf-label-image -i ~/Downloads/Parrot.red.macaw.1.arp.750pix.jpg
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
0.939399: 89:macaw
0.060436: 91:lorikeet
0.000062: 90:sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
0.000057: 24:vulture
0.000023: 88:African grey, African gray, Psittacus erithacus
time: 28.257ms
The complete source code for this example is available on Github: 
https://github.com/canonical/tf-lite-examples-snap/tree/tf-lite-blogpost/image-labelling

You can find more complex TensorFlow Lite examples packaged with snap at: https://github.com/canonical/tf-lite-examples-snap 

Delta updates
The most likely scenario of a production ML workload would be a regular update of the model. These models could be quite large, but still only a fraction of the size of the entire snap. 

Many edge devices are connected using an expensive LTE or satellite internet connection. Data costs are a huge concern in some of these cases. Taking the size of the update into account is therefore important.

Snaps can update via binary delta updates – that is, only sending the binary changes to the edge device, without having to download the entire new package. To make full use of this, one needs to make sure your snapcraft.yaml file is written in such a way that only the required parts change when you update something.

This is the reason for splitting the dependencies, the Python script, and the model and labels into separate parts. In our example, if the model gets updated, only the changes in the model will be sent to the edge device during an update. The dependencies and the Python script won’t be sent again if they do not change. Keep in mind that deltas are calculated on the compressed package, which means that the deltas aren’t equal to the changed bytes. This also depends on the selected compression algorithm for the snap.

Testing and rolling out your updates
In a production environment, you should not blindly roll out updates to all your devices without testing it properly first. Snapcraft and the Snapstore have features to assist with this. Two of the valuable snap features are Channels and Progressive Release management.

Channels are used to publish stable and pre-release versions of your software. Your production devices will be subscribed to updates from the stable channel, while your test devices will be looking for updates on the edge channel. You can have multiple channels for various levels of confidence of the version. As a new update gets tested on the edge channel, it can be moved to beta, or candidate as confidence increases, and be tested there by a wider audience.

The progressive release mechanism allows incremental update roll-outs to a predefined percentage of your devices. For example, a ratio of your testers that are subscribed to updates on the beta channel. You do not push the update to all of them at the same time. Rather start with a small percentage, say 20%. After a day, if there are no issues, you increase the percentage to 50%, and after another day to 100%. After a week of testing with no reported issues, you can promote the update to the candidate channel and follow a similar progressive release strategy. Eventually, the update can be released to the stable channel for the entire target device set.

Release management is a complex task, but the Snap Store offers various features to simplify it; read more here. As your deployments scale, it would become beneficial to rely on more powerful tools to manage updates to your fleets. Landscape offers a good collection of remote management and monitoring features.

A fully containerised solution
Since we created a snap, our TF Lite application can be easily deployed on Ubuntu Core and integrated into a software stack that is fully containerised and maintainable. The snap can be easily deployed and installed on a pre-built Ubuntu Core image, however, this isn’t what you would do in production. 

Ubuntu Core comes with tooling that makes it possible to create your production images that bundle your stack’s building blocks, including the ML software. Refer to this documentation to get started with building your Ubuntu Core image.



