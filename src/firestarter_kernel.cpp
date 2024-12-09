#include <roco2/kernels/firestarter.hpp>
#include <roco2/memory/thread_local.hpp>
#include <roco2/metrics/utility.hpp>
#include <roco2/cpu/affinity.hpp>
#include <roco2/scorep.hpp>

#include <cassert>

namespace roco2
{
namespace kernels
{

    firestarter::firestarter()
    {
        env.evaluateCpuAffinity(0, "");
        env.evaluateFunctions();
        env.selectFunction(0, false);
        env.printSelectedCodePathSummary();

        loadVar = LOAD_HIGH;

        roco2::thread_local_memory().lwd = std::make_unique<::firestarter::LoadWorkerData>( cpu::info::current_thread(), env, &loadVar, 0, false, false);
    }

    static void stop(roco2::chrono::time_point until, long long unsigned int*loadVar)
    {
        std::this_thread::sleep_for(until - std::chrono::high_resolution_clock::now());
        *loadVar = LOAD_STOP;
    }

    void firestarter::run_kernel(roco2::chrono::time_point until)
    {
        std::thread cntrl_thread;
        if(cpu::info::current_thread() == 0)
        {
            cntrl_thread = std::thread(stop, until, &loadVar);
        }

#ifdef HAS_SCOREP
        SCOREP_USER_REGION("firestarter_kernel", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
        auto&my_mem_buffer = roco2::thread_local_memory().firestarter_buffer;

        auto& lwd = roco2::thread_local_memory().lwd;

        lwd->environment().setCpuAffinity(lwd->id());
        lwd->config().payload().compilePayload(
            lwd->config().payloadSettings(), lwd->config().instructionCacheSize(),
            lwd->config().dataCacheBufferSize(), lwd->config().ramBufferSize(),
            lwd->config().thread(), lwd->config().lines(), lwd->dumpRegisters,
            lwd->errorDetection);

        auto dataCacheSizeIt =
            lwd->config().platformConfig().dataCacheBufferSize().begin();

        auto ramBufferSize = lwd->config().platformConfig().ramBufferSize();

        lwd->buffersizeMem = (*dataCacheSizeIt + *std::next(dataCacheSizeIt, 1) +
            *std::next(dataCacheSizeIt, 2) + ramBufferSize) /
            lwd->config().thread() / sizeof(unsigned long long);

        roco2::thread_local_memory().create_firestarter_buffer(lwd->buffersizeMem);
        lwd->addrMem = roco2::thread_local_memory().firestarter_buffer.data() + lwd->addrOffset;

        lwd->config().payload().init(lwd->addrMem, lwd->buffersizeMem);

        // check alignment requirements
        assert(reinterpret_cast<param_type>(my_mem_buffer.data()) % 64 == 0);

        std::size_t loops = 0;
        do
        {
#ifdef HAS_SCOREP
            // SCOREP_USER_REGION("firestarter_kernel_loop", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
            lwd->iterations = lwd->config().payload().highLoadFunction(
            lwd->addrMem, lwd->addrHigh, lwd->iterations);
            lwd->config().payload().lowLoadFunction(lwd->addrHigh, lwd->period);

        } while (std::chrono::high_resolution_clock::now() < until);

        roco2::metrics::utility::instance().write(loops);
        if(cpu::info::current_thread() == 0)
        {
            cntrl_thread.join();
        }
    }
} // namespace kernels
} // namespace roco2
