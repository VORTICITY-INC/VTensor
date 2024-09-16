Installation
==========

The project is built using Bazel 7.3.1 and has been tested with CUDA 12.XX.

Bazel installation
------------

Please follow `Bazelisk <https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation>`_ installation to install Bazelisk for Bazel version management. For linux, you could download the binary from the release page and move it to the /usr/bin directory or add the binary to your `Path`.

.. code-block:: bash

    wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
    chmod +x bazelisk-linux-amd64
    sudo mv bazelisk-linux-amd64 /usr/bin/bazel

Git clone the repository
------------

.. code-block:: bash

    git clone https://github.com/VORTICITY-INC/VTensor.git
    cd VTensor

Run Test
------------

.. code-block:: bash

    bazel test //...

Run Examples
------------

.. code-block:: bash

    bazel run //examples:monte_carlo

Include the porject in your Bazel project
------------

Please follow the Bazel `documentation <https://docs.bazel.build/versions/4.2.1/external.html#depending-on-other-bazel-projects>`_ to include the project in your Bazel project.
