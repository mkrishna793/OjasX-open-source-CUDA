// ojasx2/csrc/ojasx_openreg_guard.cpp
// Native C++ LibTorch DeviceGuardImplInterface for OjasX PrivateUse1 (OpenReg)

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

namespace ojasx {

static thread_local c10::DeviceIndex current_ojasx_device = 0;

struct OjasXGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  OjasXGuardImpl() = default;

  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    c10::Device old(c10::DeviceType::PrivateUse1, current_ojasx_device);
    current_ojasx_device = d.index() >= 0 ? d.index() : 0;
    return old;
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, current_ojasx_device);
  }

  void setDevice(c10::Device d) const override {
    current_ojasx_device = d.index() >= 0 ? d.index() : 0;
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    current_ojasx_device = d.index() >= 0 ? d.index() : 0;
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, s.device());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, OjasXGuardImpl);

} // namespace ojasx
