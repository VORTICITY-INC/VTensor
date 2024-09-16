IO operations
==============

VTensor supports reading and writing tensors from/to host memory.

From host to device
------------

.. code-block:: cpp

    #include <lib/vtensor.hpp>

    int main() {
        std::vector<float> vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        auto tensor = vt::astensor(vector);
        auto tensor1 = vt::astensor(vector.data(), vector.size());
        return 0;
    }

From device to host
------------

.. code-block:: cpp

    #include <lib/vtensor.hpp>

    int main() {
        auto tensor = vt::arange(12);
        auto vector = vt::asvector(tensor);
        return 0;
    }


Save/Load from disk
------------
VTensor employs `cnpy <https://github.com/rogersce/cnpy>`_ for disk-based save and load operations.

.. code-block:: cpp

    #include <lib/vtensor.hpp>

    int main() {
        auto tensor = vt::arange(12);
        vt::save("test.npy", tensor);
        auto tensor1 = vt::load<float>("test.npy");
        return 0;
    }
