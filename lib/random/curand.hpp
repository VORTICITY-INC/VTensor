#pragma once

#include <curand.h>

namespace vt {
namespace cuda {

// Random number generator types
struct Default {};
struct XORWOW {};
struct MMRG32K3A {};
struct MTGP32 {};
struct MT19937 {};
struct PHILOX4_32_10 {};
struct SOBOL32 {};
struct SCRAMBLED_SOBOL32 {};
struct SOBOL64 {};
struct SCRAMBLED_SOBOL64 {};

// Type traits for random number generator types
template <typename T>
struct RNGTypeTraits;

template <>
struct RNGTypeTraits<Default> {
    static constexpr curandRngType_t value = CURAND_RNG_PSEUDO_DEFAULT;
};

template <>
struct RNGTypeTraits<XORWOW> {
    static constexpr curandRngType_t value = CURAND_RNG_PSEUDO_XORWOW;
};

template <>
struct RNGTypeTraits<MMRG32K3A> {
    static constexpr curandRngType_t value = CURAND_RNG_PSEUDO_MRG32K3A;
};

template <>
struct RNGTypeTraits<MTGP32> {
    static constexpr curandRngType_t value = CURAND_RNG_PSEUDO_MTGP32;
};

template <>
struct RNGTypeTraits<MT19937> {
    static constexpr curandRngType_t value = CURAND_RNG_PSEUDO_MT19937;
};

template <>
struct RNGTypeTraits<PHILOX4_32_10> {
    static constexpr curandRngType_t value = CURAND_RNG_PSEUDO_PHILOX4_32_10;
};

template <>
struct RNGTypeTraits<SOBOL32> {
    static constexpr curandRngType_t value = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
};

template <>
struct RNGTypeTraits<SCRAMBLED_SOBOL32> {
    static constexpr curandRngType_t value = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
};

template <>
struct RNGTypeTraits<SOBOL64> {
    static constexpr curandRngType_t value = CURAND_RNG_QUASI_SOBOL64;
};

template <>
struct RNGTypeTraits<SCRAMBLED_SOBOL64> {
    static constexpr curandRngType_t value = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
};

// Alias to check the quasi-random number generator type
template <typename RNG>
using is_quasi_rg =
    std::disjunction<std::is_same<RNG, SOBOL32>, std::is_same<RNG, SOBOL64>, std::is_same<RNG, SCRAMBLED_SOBOL32>, std::is_same<RNG, SCRAMBLED_SOBOL64>>;

// Alias to check the pseudo-random number generator type
template <typename RNG>
using is_pseudo_rg = std::disjunction<std::is_same<RNG, Default>, std::is_same<RNG, XORWOW>, std::is_same<RNG, MMRG32K3A>, std::is_same<RNG, MTGP32>,
                                      std::is_same<RNG, MT19937>, std::is_same<RNG, PHILOX4_32_10>>;

/**
 * @brief check the status of the CuRand library.
 *
 * @param status: The status of the CuRand library.
 * @param message: The message to be displayed if the status is not CURAND_STATUS_SUCCESS.
 * @throw std::runtime_error: If the status is not CURAND_STATUS_SUCCESS.
 */
inline void check_curand_status(curandStatus_t status, const std::string& message) {
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error(message + ": " + std::to_string(status));
    }
}

/**
 * @brief Deleter for the CuRand handle.
 */
struct CuRandHandleDeleter {
    void operator()(curandGenerator_t* handle) const {
        if (handle) {
            check_curand_status(curandDestroyGenerator(*handle), "Failed to destroy CuRand handle");
            delete handle;
        }
    }
};

// Alias for the CuRand unique pointer handle
using CuRandHandleT = std::unique_ptr<curandGenerator_t, CuRandHandleDeleter>;

/**
 * @brief Create a CuRand unique pointer handle.
 *
 * @tparam RNG: The random number generator type.
 * @param dim: The dimension for the quasi-random number generator.
 * @return CuRandHandleT: The CuRand unique pointer handle.
 */
template <typename RNG = Default>
CuRandHandleT create_curand_handle(size_t dim = 1) {
    auto handle = CuRandHandleT(new curandGenerator_t);
    check_curand_status(curandCreateGenerator(handle.get(), RNGTypeTraits<RNG>::value), "Failed to create CuRand handle");
    if constexpr (is_quasi_rg<RNG>::value) {
        check_curand_status(curandSetQuasiRandomGeneratorDimensions(*handle, dim), "Failed to set quasi-random generator dimensions");
    }
    return handle;
}

// Singleton class for global CuRand handle management
class CuRand {
   public:
    // Delete copy constructor and assignment operator to ensure Singleton
    CuRand(const CuRand&) = delete;
    CuRand& operator=(const CuRand&) = delete;

    /**
     * @brief Get the Instance object. It returns the singleton instance of the CuRand handle.
     *
     * @return CuRand&
     */
    static CuRand& get_instance() {
        static CuRand instance;
        return instance;
    }

    /**
     * @brief Get the CuRand handle.
     *
     * @return curandGenerator_t: The CuRand handle.
     */
    curandGenerator_t get_handle() const { return *handle.get(); }

   private:
    // Private constructor to enforce Singleton
    CuRand() { handle = create_curand_handle(); }

    // Unique pointer to manage the CuRand handle
    CuRandHandleT handle;
};

/**
 * @brief Set the seed for the pseudo random number generator.
 *
 * @param seed: The seed for the pseudo random number generator.
 * @param gen: The CuRand handle. The default is the global CuRand handle.
 */
inline void set_seed(size_t seed, curandGenerator_t gen = CuRand::get_instance().get_handle()) {
    check_curand_status(curandSetPseudoRandomGeneratorSeed(gen, seed), "Failed to set pseudo-random generator seed");
}

/**
 * @brief Set the offset for the pseudo/quasi random number generator.
 *
 * @param offset: The offset for the pseudo/quasi random number generator.
 * @param gen: The CuRand handle. The default is the global CuRand handle.
 */
inline void set_offset(size_t offset, curandGenerator_t gen = CuRand::get_instance().get_handle()) {
    check_curand_status(curandSetGeneratorOffset(gen, offset), "Failed to set generator offset");
}

}  // namespace cuda

}  // namespace vt
