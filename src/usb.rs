use nusb::descriptors::TransferType;
use nusb::transfer::{Bulk, Direction, In};
use nusb::{DeviceInfo, Interface, MaybeFuture};
use thiserror::Error;

pub const TRACE_PRODUCT: &str = "FOCI OpenFFBoard (trace)";
pub const TRACE_INTERFACE_CLASS: u8 = 0xff;

#[derive(Debug, Error)]
pub enum UsbTraceError {
    #[error("USB error: {0}")]
    Nusb(#[from] nusb::Error),
    #[error("trace endpoint not found for device {0}")]
    TraceEndpointMissing(String),
    #[error("trace interface not found for device {0}")]
    TraceInterfaceMissing(String),
}

#[derive(Debug, Clone)]
pub struct TraceDeviceInfo {
    pub path: String,
    pub product: String,
    pub serial: String,
}

impl TraceDeviceInfo {
    pub fn path(&self) -> &str {
        &self.path
    }
}

pub struct OpenTraceDevice {
    pub info: TraceDeviceInfo,
    pub interface_number: u8,
    pub endpoint_address: u8,
    pub max_packet_size: usize,
    interface: Interface,
}

impl OpenTraceDevice {
    pub fn bulk_reader(
        &self,
        transfer_size: usize,
        read_timeout: std::time::Duration,
    ) -> Result<nusb::io::EndpointRead<Bulk>, UsbTraceError> {
        let endpoint = self.interface.endpoint::<Bulk, In>(self.endpoint_address)?;
        Ok(endpoint
            .reader(transfer_size)
            .with_num_transfers(4)
            .with_read_timeout(read_timeout))
    }
}

pub fn find_trace_devices(serial_filter: &[String]) -> Result<Vec<OpenTraceDevice>, UsbTraceError> {
    let devices = nusb::list_devices().wait()?;
    let mut opened = Vec::new();

    for device in devices {
        if let Some(trace_device) = open_if_trace_device(device, serial_filter)? {
            opened.push(trace_device);
        }
    }

    Ok(opened)
}

fn open_if_trace_device(
    device: DeviceInfo,
    serial_filter: &[String],
) -> Result<Option<OpenTraceDevice>, UsbTraceError> {
    let product = device.product_string().unwrap_or_default().to_string();
    if product != TRACE_PRODUCT {
        return Ok(None);
    }

    let serial = device.serial_number().unwrap_or_default().to_string();
    if !serial_filter.is_empty() && !serial_filter.iter().any(|value| value == &serial) {
        return Ok(None);
    }

    let path = trace_device_path(&device);
    let interface_number = device
        .interfaces()
        .find(|interface| interface.class() == TRACE_INTERFACE_CLASS)
        .map(|interface| interface.interface_number())
        .ok_or_else(|| UsbTraceError::TraceInterfaceMissing(path.clone()))?;
    let open_device = device.open().wait()?;
    let interface = open_device.claim_interface(interface_number).wait()?;
    let (endpoint_address, max_packet_size) = find_trace_endpoint(&interface, &path)?;

    Ok(Some(OpenTraceDevice {
        info: TraceDeviceInfo {
            path,
            product,
            serial,
        },
        interface_number,
        endpoint_address,
        max_packet_size,
        interface,
    }))
}

fn find_trace_endpoint(interface: &Interface, path: &str) -> Result<(u8, usize), UsbTraceError> {
    let Some(descriptor) = interface.descriptor() else {
        return Err(UsbTraceError::TraceEndpointMissing(path.to_string()));
    };
    for endpoint in descriptor.endpoints() {
        if endpoint.direction() == Direction::In && endpoint.transfer_type() == TransferType::Bulk {
            return Ok((endpoint.address(), endpoint.max_packet_size()));
        }
    }

    Err(UsbTraceError::TraceEndpointMissing(path.to_string()))
}

fn trace_device_path(device: &DeviceInfo) -> String {
    format!("{}-{}", device.bus_id(), device.device_address())
}
