//! The DXIL validator in `dxil.dll` (`IDxcValidator`), for tests and the
//! offline batch. Translated shaders do not need it: they carry the
//! container hash we compute (`container::hash`).
//!
//! The DLL is found at `VICTORY_DXIL` if set, else the newest Windows SDK's
//! `bin\<version>\x64\dxil.dll`, else wherever `LoadLibrary` finds it.

use std::ffi::{OsStr, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::PathBuf;

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

type Hresult = i32;

#[repr(C)]
struct Guid(u32, u16, u16, [u8; 8]);

const CLSID_DXC_VALIDATOR: Guid = Guid(
    0x8ca3_e215,
    0xf728,
    0x4cf3,
    [0x8c, 0xdd, 0x88, 0xaf, 0x91, 0x75, 0x87, 0xa1],
);
const IID_IDXC_VALIDATOR: Guid = Guid(
    0xa6e8_2bd2,
    0x1fd7,
    0x4826,
    [0x98, 0x11, 0x28, 0x57, 0xe7, 0x97, 0xf4, 0x9a],
);
/// `DxcValidatorFlags_InPlaceEdit`: sign the container we pass.
const IN_PLACE_EDIT: u32 = 1;

type CreateInstance =
    unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> Hresult;
type FreeFn = unsafe extern "system" fn(HMODULE) -> i32;

/// COM object: a pointer to a table of method pointers.
type Object = *mut *const usize;

unsafe fn method<F: Copy>(obj: Object, slot: usize) -> F {
    // SAFETY: the caller names a slot of the object's interface.
    unsafe { std::mem::transmute_copy(&*(*obj).add(slot)) }
}

unsafe fn release(obj: Object) {
    if !obj.is_null() {
        // SAFETY: slot 2 of every COM interface is Release.
        unsafe { method::<unsafe extern "system" fn(Object) -> u32>(obj, 2)(obj) };
    }
}

/// An `IDxcBlob` over our bytes: QueryInterface, AddRef, Release,
/// GetBufferPointer, GetBufferSize. Lives on the caller's stack.
#[repr(C)]
struct Blob {
    vtable: *const BlobVtable,
    data: *mut u8,
    len: usize,
}

/// IUnknown and IDxcBlob, as raw GUID bytes.
const IUNKNOWN: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0xc0, 0, 0, 0, 0, 0, 0, 0x46];
const IDXC_BLOB: [u8; 16] = [
    0x08, 0xfb, 0xa5, 0x8b, 0x95, 0x51, 0xe2, 0x40, 0xac, 0x58, 0x0d, 0x98, 0x9c, 0x3a, 0x01, 0x02,
];

unsafe extern "system" fn blob_qi(
    this: *mut Blob,
    iid: *const [u8; 16],
    out: *mut *mut c_void,
) -> Hresult {
    // SAFETY: the validator passes valid pointers.
    unsafe {
        if *iid == IUNKNOWN || *iid == IDXC_BLOB {
            *out = this.cast();
            0
        } else {
            *out = std::ptr::null_mut();
            0x8000_4002_u32 as i32 // E_NOINTERFACE
        }
    }
}
unsafe extern "system" fn blob_ref(_: *mut Blob) -> u32 {
    1
}
unsafe extern "system" fn blob_ptr(this: *mut Blob) -> *mut u8 {
    // SAFETY: `this` is our blob.
    unsafe { (*this).data }
}
unsafe extern "system" fn blob_len(this: *mut Blob) -> usize {
    // SAFETY: `this` is our blob.
    unsafe { (*this).len }
}

#[repr(C)]
struct BlobVtable {
    qi: unsafe extern "system" fn(*mut Blob, *const [u8; 16], *mut *mut c_void) -> Hresult,
    add_ref: unsafe extern "system" fn(*mut Blob) -> u32,
    release: unsafe extern "system" fn(*mut Blob) -> u32,
    ptr: unsafe extern "system" fn(*mut Blob) -> *mut u8,
    len: unsafe extern "system" fn(*mut Blob) -> usize,
}

static BLOB_VTABLE: BlobVtable = BlobVtable {
    qi: blob_qi,
    add_ref: blob_ref,
    release: blob_ref,
    ptr: blob_ptr,
    len: blob_len,
};

struct Module {
    handle: HMODULE,
    free: FreeFn,
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: `Module` owns one loaded library reference.
        unsafe { (self.free)(self.handle) };
    }
}

pub struct Validator {
    object: Object,
    // Dropped after `Validator::drop` releases `object`, so COM vtable calls
    // never execute against an unmapped DLL.
    _module: Option<Module>,
}

// SAFETY: the validator object is free-threaded (DXC's validator has no
// thread affinity); calls are serialised by `&mut self`, and process module
// handles can be freed from any thread.
unsafe impl Send for Validator {}

impl Validator {
    /// Where `dxil.dll` is looked for.
    pub fn default_path() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("VICTORY_DXIL") {
            return Some(p.into());
        }
        let kits = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
        let mut versions: Vec<PathBuf> = std::fs::read_dir(&kits)
            .ok()?
            .filter_map(Result::ok)
            .map(|e| e.path().join(r"x64\dxil.dll"))
            .filter(|p| p.exists())
            .collect();
        versions.sort();
        versions.pop()
    }

    /// Loads `path`'s validator (or `dxil.dll` from the DLL search path).
    pub fn load(path: Option<&std::path::Path>) -> Result<Self, String> {
        let target = path.map_or_else(|| OsStr::new("dxil.dll"), std::path::Path::as_os_str);
        let name: Vec<u16> = target.encode_wide().chain([0]).collect();
        // SAFETY: plain Win32 calls with a NUL-terminated name.
        unsafe {
            let module = LoadLibraryW(name.as_ptr());
            if module.is_null() {
                return Err(format!("cannot load {}", target.to_string_lossy()));
            }
            Self::from_module(module, FreeLibrary, |module| {
                let create = GetProcAddress(module, c"DxcCreateInstance".as_ptr().cast())
                    .ok_or("dxil.dll lacks DxcCreateInstance")?;
                let create: CreateInstance = std::mem::transmute(create);
                let mut object = std::ptr::null_mut();
                let hr = create(&CLSID_DXC_VALIDATOR, &IID_IDXC_VALIDATOR, &mut object);
                if hr < 0 || object.is_null() {
                    release(object.cast());
                    return Err(format!("DxcCreateInstance(validator) failed: {hr:#x}"));
                }
                Ok(object.cast())
            })
        }
    }

    // SAFETY: `module` is one owned handle accepted by `free`; on `Ok`,
    // `create` returns one owned validator interface backed by `module`.
    unsafe fn from_module(
        module: HMODULE,
        free: FreeFn,
        create: impl FnOnce(HMODULE) -> Result<Object, String>,
    ) -> Result<Self, String> {
        let module = Module {
            handle: module,
            free,
        };
        Ok(Validator {
            object: create(module.handle)?,
            _module: Some(module),
        })
    }

    /// Validates `container` and signs it in place; the error text if it
    /// is rejected.
    pub fn validate(&mut self, container: &mut [u8]) -> Result<(), String> {
        if container.starts_with(b"DXBC") {
            crate::dxbc::Container::parse(container).map_err(|e| e.to_string())?;
        }
        let mut blob = Blob {
            vtable: &BLOB_VTABLE,
            data: container.as_mut_ptr(),
            len: container.len(),
        };
        // SAFETY: IDxcValidator::Validate (slot 3) with our blob; then
        // IDxcOperationResult GetStatus (3) and GetErrorBuffer (5), and
        // IDxcBlob GetBufferPointer (3) / GetBufferSize (4).
        unsafe {
            let mut result: Object = std::ptr::null_mut();
            let validate: unsafe extern "system" fn(
                Object,
                *mut Blob,
                u32,
                *mut Object,
            ) -> Hresult = method(self.object, 3);
            let hr = validate(self.object, &mut blob, IN_PLACE_EDIT, &mut result);
            if hr < 0 || result.is_null() {
                release(result);
                return Err(format!("Validate failed: {hr:#x}"));
            }
            let mut status = 0;
            let get_status: unsafe extern "system" fn(Object, *mut Hresult) -> Hresult =
                method(result, 3);
            let status_hr = get_status(result, &mut status);
            if status_hr < 0 {
                release(result);
                return Err(format!("GetStatus failed: {status_hr:#x}"));
            }
            let mut text = String::new();
            if status < 0 {
                let mut errors: Object = std::ptr::null_mut();
                let get_errors: unsafe extern "system" fn(Object, *mut Object) -> Hresult =
                    method(result, 5);
                if get_errors(result, &mut errors) >= 0 && !errors.is_null() {
                    let ptr: unsafe extern "system" fn(Object) -> *const u8 = method(errors, 3);
                    let len: unsafe extern "system" fn(Object) -> usize = method(errors, 4);
                    let n = len(errors);
                    if n > 0 {
                        let p = ptr(errors);
                        if !p.is_null() {
                            let bytes = std::slice::from_raw_parts(p, n);
                            text = String::from_utf8_lossy(bytes)
                                .trim_end_matches('\0')
                                .trim()
                                .into();
                        }
                    }
                    release(errors);
                }
                if text.is_empty() {
                    text = format!("rejected ({status:#x})");
                }
            }
            release(result);
            if status < 0 { Err(text) } else { Ok(()) }
        }
    }
}

impl Drop for Validator {
    fn drop(&mut self) {
        // SAFETY: we own one reference, released before `_module` unloads.
        unsafe { release(self.object) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E_FAIL: Hresult = 0x8000_4005_u32 as Hresult;
    const E_INVALIDARG: Hresult = 0x8007_0057_u32 as Hresult;

    #[repr(C)]
    struct Stub<T> {
        vtable: *const usize,
        releases: u32,
        inner: T,
    }

    impl<T> Stub<T> {
        fn new(vtable: &[usize], inner: T) -> Self {
            Self {
                vtable: vtable.as_ptr(),
                releases: 0,
                inner,
            }
        }

        fn as_object(&mut self) -> Object {
            std::ptr::from_mut(self).cast()
        }
    }

    unsafe extern "system" fn stub_release(this: Object) -> u32 {
        // SAFETY: every stub begins with `(vtable, releases)`.
        unsafe {
            let releases = &mut (*this.cast::<Stub<()>>()).releases;
            *releases += 1;
            0
        }
    }

    struct ValidatorStub {
        hr: Hresult,
        result: Object,
    }

    struct ResultStub {
        status_hr: Hresult,
        status: Hresult,
        errors_hr: Hresult,
        errors: Object,
    }

    struct ErrorBlobStub {
        bytes: &'static [u8],
        null_ptr: bool,
        ptr_calls: u32,
    }

    unsafe extern "system" fn stub_validate(
        this: Object,
        blob: *mut Blob,
        flags: u32,
        out: *mut Object,
    ) -> Hresult {
        // SAFETY: invoked on `Stub<ValidatorStub>` with a live `Blob`.
        unsafe {
            let stub = &(*this.cast::<Stub<ValidatorStub>>()).inner;
            let vt = &*(*blob).vtable;
            assert_eq!(flags, IN_PLACE_EDIT);
            if (vt.len)(blob) > 0 {
                *(vt.ptr)(blob) ^= 0xff;
            }
            *out = stub.result;
            stub.hr
        }
    }

    unsafe extern "system" fn stub_get_status(this: Object, out: *mut Hresult) -> Hresult {
        // SAFETY: invoked on `Stub<ResultStub>`.
        unsafe {
            let stub = &(*this.cast::<Stub<ResultStub>>()).inner;
            if stub.status_hr >= 0 {
                *out = stub.status;
            }
            stub.status_hr
        }
    }

    unsafe extern "system" fn stub_get_errors(this: Object, out: *mut Object) -> Hresult {
        // SAFETY: invoked on `Stub<ResultStub>`.
        unsafe {
            let stub = &(*this.cast::<Stub<ResultStub>>()).inner;
            if stub.errors_hr >= 0 {
                *out = stub.errors;
            }
            stub.errors_hr
        }
    }

    unsafe extern "system" fn stub_err_ptr(this: Object) -> *const u8 {
        // SAFETY: invoked on `Stub<ErrorBlobStub>`.
        unsafe {
            let stub = &mut (*this.cast::<Stub<ErrorBlobStub>>()).inner;
            stub.ptr_calls += 1;
            if stub.null_ptr || stub.bytes.is_empty() {
                std::ptr::null()
            } else {
                stub.bytes.as_ptr()
            }
        }
    }

    unsafe extern "system" fn stub_err_len(this: Object) -> usize {
        // SAFETY: invoked on `Stub<ErrorBlobStub>`.
        let stub = unsafe { &(*this.cast::<Stub<ErrorBlobStub>>()).inner };
        stub.bytes.len()
    }

    #[test]
    fn get_status_failure_is_not_treated_as_valid() {
        let rel = stub_release as *const () as usize;
        let res_vt = [0, 0, rel, stub_get_status as *const () as usize, 0, 0];
        let val_vt = [0, 0, rel, stub_validate as *const () as usize];
        let mut res = Stub::new(
            &res_vt,
            ResultStub {
                status_hr: E_FAIL,
                status: 0,
                errors_hr: 0,
                errors: std::ptr::null_mut(),
            },
        );
        let mut val = Stub::new(
            &val_vt,
            ValidatorStub {
                hr: 0,
                result: res.as_object(),
            },
        );
        let mut data = [0xaa, 0xbb];
        let outcome = Validator {
            object: val.as_object(),
            _module: None,
        }
        .validate(&mut data);
        assert_eq!(outcome, Err("GetStatus failed: 0x80004005".to_owned()));
        assert_eq!(res.releases, 1);
        assert_eq!(val.releases, 1);
    }

    #[test]
    fn empty_or_null_diagnostic_buffer_avoids_raw_slice_ub() {
        let rel = stub_release as *const () as usize;
        let err_vt = [
            0,
            0,
            rel,
            stub_err_ptr as *const () as usize,
            stub_err_len as *const () as usize,
        ];
        let res_vt = [
            0,
            0,
            rel,
            stub_get_status as *const () as usize,
            0,
            stub_get_errors as *const () as usize,
        ];
        let val_vt = [0, 0, rel, stub_validate as *const () as usize];

        for (bytes, null_ptr, expected_ptr_calls) in [(&b""[..], true, 0), (&b"oops"[..], true, 1)]
        {
            let mut errs = Stub::new(
                &err_vt,
                ErrorBlobStub {
                    bytes,
                    null_ptr,
                    ptr_calls: 0,
                },
            );
            let mut res = Stub::new(
                &res_vt,
                ResultStub {
                    status_hr: 0,
                    status: E_INVALIDARG,
                    errors_hr: 0,
                    errors: errs.as_object(),
                },
            );
            let mut val = Stub::new(
                &val_vt,
                ValidatorStub {
                    hr: 0,
                    result: res.as_object(),
                },
            );
            let mut data = [1u8];
            let outcome = Validator {
                object: val.as_object(),
                _module: None,
            }
            .validate(&mut data);
            assert_eq!(outcome, Err("rejected (0x80070057)".to_owned()));
            assert_eq!(errs.inner.ptr_calls, expected_ptr_calls);
            assert_eq!(errs.releases, 1);
            assert_eq!(res.releases, 1);
        }
    }

    #[test]
    fn rejection_trims_diagnostic_and_in_place_success_edits_container() {
        let rel = stub_release as *const () as usize;
        let err_vt = [
            0,
            0,
            rel,
            stub_err_ptr as *const () as usize,
            stub_err_len as *const () as usize,
        ];
        let res_vt = [
            0,
            0,
            rel,
            stub_get_status as *const () as usize,
            0,
            stub_get_errors as *const () as usize,
        ];
        let val_vt = [0, 0, rel, stub_validate as *const () as usize];

        let mut errs = Stub::new(
            &err_vt,
            ErrorBlobStub {
                bytes: b"  missing PSV0\r\n\0",
                null_ptr: false,
                ptr_calls: 0,
            },
        );
        let mut fail_res = Stub::new(
            &res_vt,
            ResultStub {
                status_hr: 0,
                status: E_FAIL,
                errors_hr: 0,
                errors: errs.as_object(),
            },
        );
        let mut fail_val = Stub::new(
            &val_vt,
            ValidatorStub {
                hr: 0,
                result: fail_res.as_object(),
            },
        );
        let mut buf = [0x10];
        let err = Validator {
            object: fail_val.as_object(),
            _module: None,
        }
        .validate(&mut buf);
        assert_eq!(err, Err("missing PSV0".to_owned()));
        assert_eq!(errs.releases, 1);
        assert_eq!(fail_res.releases, 1);

        let mut ok_res = Stub::new(
            &res_vt,
            ResultStub {
                status_hr: 0,
                status: 0,
                errors_hr: 0,
                errors: std::ptr::null_mut(),
            },
        );
        let mut ok_val = Stub::new(
            &val_vt,
            ValidatorStub {
                hr: 0,
                result: ok_res.as_object(),
            },
        );
        let mut signed = [0x0f, 0x22];
        assert_eq!(
            Validator {
                object: ok_val.as_object(),
                _module: None,
            }
            .validate(&mut signed),
            Ok(())
        );
        assert_eq!(signed, [0xf0, 0x22]);
        assert_eq!(ok_res.releases, 1);
    }

    #[test]
    fn blob_query_interface_accepts_only_known_iids() {
        let mut data = [7u8];
        let mut blob = Blob {
            vtable: &BLOB_VTABLE,
            data: data.as_mut_ptr(),
            len: data.len(),
        };
        let ptr = std::ptr::from_mut(&mut blob);
        // SAFETY: live stack `Blob` and valid output pointer.
        unsafe {
            let mut out = std::ptr::null_mut();
            assert_eq!((BLOB_VTABLE.qi)(ptr, &IUNKNOWN, &mut out), 0);
            assert_eq!(out, ptr.cast());
            assert_eq!((BLOB_VTABLE.qi)(ptr, &IDXC_BLOB, &mut out), 0);
            assert_eq!(out, ptr.cast());
            assert_eq!(
                (BLOB_VTABLE.qi)(ptr, &[0xaa; 16], &mut out),
                0x8000_4002_u32 as Hresult
            );
            assert!(out.is_null());
        }
    }

    #[derive(Default)]
    struct ModuleProbe {
        object: Object,
        frees: u32,
        releases_before_free: u32,
    }

    unsafe extern "system" fn probe_free(handle: HMODULE) -> i32 {
        // SAFETY: `handle` points to a live `ModuleProbe`.
        unsafe {
            let probe = &mut *handle.cast::<ModuleProbe>();
            probe.frees += 1;
            if !probe.object.is_null() {
                probe.releases_before_free = (*probe.object.cast::<Stub<()>>()).releases;
            }
        }
        1
    }

    #[test]
    fn module_is_freed_on_creation_error_and_after_com_release_on_drop() {
        let mut failed_probe = ModuleProbe::default();
        let handle = std::ptr::from_mut(&mut failed_probe).cast();
        // SAFETY: `probe_free` accepts `handle`.
        let err = unsafe { Validator::from_module(handle, probe_free, |_| Err("missing".into())) };
        assert_eq!(err.err().as_deref(), Some("missing"));
        assert_eq!(failed_probe.frees, 1);

        let vt = [0, 0, stub_release as *const () as usize];
        let mut obj = Stub::new(&vt, ());
        let raw_obj = obj.as_object();
        let mut drop_probe = ModuleProbe {
            object: raw_obj,
            frees: 0,
            releases_before_free: 0,
        };
        let handle = std::ptr::from_mut(&mut drop_probe).cast();
        // SAFETY: `obj`, `vt` and `drop_probe` outlive `v`.
        let v = unsafe { Validator::from_module(handle, probe_free, |_| Ok(raw_obj)) }.unwrap();
        assert_eq!(drop_probe.frees, 0);
        drop(v);
        assert_eq!(drop_probe.frees, 1);
        assert_eq!(
            drop_probe.releases_before_free, 1,
            "COM object must be released before FreeLibrary unmaps the module"
        );
    }

    #[test]
    fn load_missing_library_omits_trailing_nul_and_real_dxil_validates() {
        let missing = std::path::Path::new(r"Z:\nonexistent-victory-dxil-1049.dll");
        let err = Validator::load(Some(missing)).err().unwrap();
        assert!(!err.contains('\0'), "unexpected NUL in error: {err:?}");

        let Ok(mut real) = Validator::load(Validator::default_path().as_deref()) else {
            return;
        };
        let mut s = super::super::shader::Shader::new(true);
        let entry = s.entry;
        s.m.terminate(entry, super::super::bitcode::Term::Ret);
        let mut valid = s.finish();
        valid[4..20].fill(0);
        real.validate(&mut valid).unwrap();
        assert_ne!(&valid[4..20], &[0u8; 16]);

        let mut truncated = b"DXBCtruncated-header-body!!".to_vec();
        assert!(real.validate(&mut truncated).is_err());

        let mut broken = super::super::container::build(&[(b"SFI0", &[0u8; 8])]);
        assert!(real.validate(&mut broken).is_err());
    }
}
