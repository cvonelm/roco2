#ifndef INCLUDE_ROCO2_KERNELS_FIRESTARTER_HPP
#define INCLUDE_ROCO2_KERNELS_FIRESTARTER_HPP

#include <roco2/kernels/base_kernel.hpp>

#include <roco2/chrono/util.hpp>

#ifdef __x86_64__
#include <firestarter/Environment/X86/X86Environment.hpp>
#elif __aarch64__
#include <firestarter/Environment/AArch64/AArch64Environment.cpp>
#else
#error "roco2 not implemented for this architecture!"
#endif
namespace roco2
{
namespace kernels
{

    class firestarter : public base_kernel
    {

        using param_type = unsigned long long;

    public:
        firestarter();

        virtual experiment_tag tag() const override
        {
            return 6;
        }

    private:
        void run_kernel(roco2::chrono::time_point until) override;
#ifdef __x86_64__
        ::firestarter::environment::x86::X86Environment env;
#elif __aarch64__
        ::firestarter::environment::aarch64::AArch64Environment env;
#endif
        long long unsigned int loadVar;
    };
}
}

#endif // INCLUDE_ROCO2_KERNELS_FIRESTARTER_HPP
