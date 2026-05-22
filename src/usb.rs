use rusb::{Device, DeviceDescriptor, DeviceHandle, GlobalContext};
use thiserror::Error;

pub const TRACE_PRODUCT: &str = "FOCI OpenFFBoard (trace)";
pub const TRACE_INTERFACE_CLASS: u8 = 0xff;

#[derive(Debug, Error)]
pub enum UsbTraceError {
    #[error("libusb error: {0}")]
    Rusb(#[from] rusb::Error),
    #[error("trace endpoint not found for device {0}")]
    TraceEndpointMissing(String),
}

#[derive(Debug, Clone)]
pub struct TraceDeviceInfo {
    pub bus: u8,
    pub address: u8,
    pub product: String,
    pub serial: String,
}

impl TraceDeviceInfo {
    pub fn path(&self) -> String {
        format!("{}-{}", self.bus, self.address)
    }
}

pub struct OpenTraceDevice {
    pub info: TraceDeviceInfo,
    pub interface_number: u8,
    pub endpoint_address: u8,
    pub max_packet_size: u16,
    pub handle: DeviceHandle<GlobalContext>,
}

pub fn find_trace_devices(serial_filter: &[String]) -> Result<Vec<OpenTraceDevice>, UsbTraceError> {
    let devices = rusb::devices()?;
    let mut opened = Vec::new();

    for device in devices.iter() {
        if let Some(trace_device) = open_if_trace_device(device, serial_filter)? {
            opened.push(trace_device);
        }
    }

    Ok(opened)
}

fn open_if_trace_device(
    device: Device<GlobalContext>,
    serial_filter: &[String],
) -> Result<Option<OpenTraceDevice>, UsbTraceError> {
    let descriptor = device.device_descriptor()?;
    let handle = device.open()?;
    let product = handle
        .read_product_string_ascii(&descriptor)
        .unwrap_or_default();
    if product != TRACE_PRODUCT {
        return Ok(None);
    }

    let serial = handle
        .read_serial_number_string_ascii(&descriptor)
        .unwrap_or_default();
    if !serial_filter.is_empty() && !serial_filter.iter().any(|value| value == &serial) {
        return Ok(None);
    }

    let (interface_number, endpoint_address, max_packet_size) =
        find_trace_endpoint(&device, &descriptor)?;
    handle.claim_interface(interface_number)?;

    Ok(Some(OpenTraceDevice {
        info: TraceDeviceInfo {
            bus: device.bus_number(),
            address: device.address(),
            product,
            serial,
        },
        interface_number,
        endpoint_address,
        max_packet_size,
        handle,
    }))
}

fn find_trace_endpoint(
    device: &Device<GlobalContext>,
    descriptor: &DeviceDescriptor,
) -> Result<(u8, u8, u16), UsbTraceError> {
    for config_index in 0..descriptor.num_configurations() {
        let config = device.config_descriptor(config_index)?;
        for interface in config.interfaces() {
            for interface_descriptor in interface.descriptors() {
                if interface_descriptor.class_code() != TRACE_INTERFACE_CLASS {
                    continue;
                }

                for endpoint in interface_descriptor.endpoint_descriptors() {
                    if endpoint.direction() == rusb::Direction::In
                        && endpoint.transfer_type() == rusb::TransferType::Bulk
                    {
                        return Ok((
                            interface_descriptor.interface_number(),
                            endpoint.address(),
                            endpoint.max_packet_size(),
                        ));
                    }
                }
            }
        }
    }

    Err(UsbTraceError::TraceEndpointMissing(format!(
        "{}-{}",
        device.bus_number(),
        device.address()
    )))
}
