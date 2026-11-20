#include "coral_registration_compat.h"  // Imports the adapter for runtime callback registrations.
#include <cassert>  // Provides assertions for memory bounds and callback fields.
#include <cstdio>  // Provides the terminal output function.
#include <sys/mman.h>  // Provides mapped memory and guard pages for the bounds check.
#include <unistd.h>  // Provides the system page size.

static void* init(TfLiteContext*, const char*, size_t) { return nullptr; }  // Supply a minimal mock initialization callback.
static void release(TfLiteContext*, void*) {}  // Supply a no-op cleanup callback for the mock operation.
static TfLiteStatus prepare(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }  // Let the mock prepare callback report success.
static TfLiteStatus invoke(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }  // Let the mock invoke callback report success.

int main() {  // Check that the compatibility adapter reads only the callback prefix.
    const long page = sysconf(_SC_PAGESIZE);  // Query the memory protection granularity of this system.
    assert(page > 0);  // Require a valid page size before allocating.
    auto* memory = static_cast<char*>(mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE,  // Request a read-write mapping spanning two pages.
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));  // Back the mapping with anonymous private memory.
    assert(memory != MAP_FAILED);  // Stop before accessing memory if mapping failed.
    assert(mprotect(memory + page, page, PROT_NONE) == 0);  // Make the second page unreadable to catch an oversized copy.
    const rps_coral::OpCallbacks callbacks{init, release, prepare, invoke};  // Build the known prefix shared by the runtime registrations.
    // Put just the known callback prefix against an unreadable page. Any
    // attempt to read the larger registration struct will fault here.
    char* legacy = memory + page - sizeof(callbacks);  // Start the prefix so its end touches the guard page.
    std::memcpy(legacy, &callbacks, sizeof(callbacks));  // Copy only the known fields into the synthetic runtime memory.
    const auto registration = rps_coral::adaptRegistration(legacy);  // Adapt this shorter prefix to the headers used by the build.
    assert(registration.init == init && registration.free == release);  // Verify the lifecycle callback pointers were copied exactly.
    assert(registration.prepare == prepare && registration.invoke == invoke);  // Verify the operation callback pointers were copied exactly.
    assert(registration.profiling_string == nullptr);  // Verify optional profiling metadata starts empty.
    assert(registration.custom_name == nullptr);  // Verify the adapter did not invent a custom operation name.
    assert(registration.builtin_code == 0 && registration.version == 0);  // Verify numerical metadata is initialized to zero.
#ifdef RPS_TEST_MODERN_REGISTRATION  // Check fields exposed by newer TensorFlow Lite headers when requested.
    assert(registration.registration_external == nullptr);  // Verify the external registration field starts empty.
    assert(registration.async_kernel == nullptr);  // Verify no asynchronous kernel was attached.
    assert(registration.inplace_operator == 0);  // Verify no in-place operator flags were set.
#endif  // End the checks that depend on the newer structure.
    assert(registration.invoke(nullptr, nullptr) == kTfLiteOk);  // Confirm the adapted invocation pointer remains callable.
    const auto empty = rps_coral::adaptRegistration(nullptr);  // Exercise the adapter when the runtime supplies no registration.
    assert(!empty.init && !empty.free && !empty.prepare && !empty.invoke);  // Verify a null runtime input produces an empty callback set.
    assert(munmap(memory, 2 * page) == 0);  // Release the test mapping and confirm cleanup succeeded.
    std::puts("PASS: callback copy stays in bounds; registration fields initialized");  // Report that the callback prefix and optional fields passed.
}  // Finish before any additional memory access.
