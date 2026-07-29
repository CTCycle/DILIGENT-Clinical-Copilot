#[cfg(windows)]
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE},
    System::{
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        },
        Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_SET_QUOTA, PROCESS_TERMINATE},
    },
};

#[cfg(windows)]
pub struct WindowsJob {
    handle: HANDLE,
}

#[cfg(windows)]
impl WindowsJob {
    pub fn attach(child: &std::process::Child) -> Result<Self, String> {
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() || handle == INVALID_HANDLE_VALUE {
                return Err("unable to create Windows Job Object".into());
            }
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _,
                std::mem::size_of_val(&limits) as u32,
            ) == 0
            {
                CloseHandle(handle);
                return Err("unable to configure Windows Job Object".into());
            }
            let process = OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_INFORMATION,
                0,
                child.id(),
            );
            if process.is_null() || AssignProcessToJobObject(handle, process) == 0 {
                if !process.is_null() {
                    CloseHandle(process);
                }
                CloseHandle(handle);
                return Err("unable to attach backend to Windows Job Object".into());
            }
            CloseHandle(process);
            Ok(Self { handle })
        }
    }
}

#[cfg(windows)]
unsafe impl Send for WindowsJob {}

#[cfg(windows)]
unsafe impl Sync for WindowsJob {}

#[cfg(windows)]
impl Drop for WindowsJob {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

#[cfg(not(windows))]
pub struct WindowsJob;

#[cfg(not(windows))]
impl WindowsJob {
    pub fn attach(_child: &std::process::Child) -> Result<Self, String> {
        Ok(Self)
    }
}
