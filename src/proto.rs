//! A minimal protobuf reader.
//!
//! Every AC EVO asset - mesh, material, design, animation - is protobuf with no
//! schema shipped, so this walks fields by number and wire type and lets the
//! caller decide what they mean. Field meanings live in load.rs next to the
//! code that uses them.
//!
//! ⚠ Every read is bounds-checked and a malformed field ends the walk rather
//! than panicking. These walkers get pointed at arbitrary bytes - a material's
//! nested blobs are not always protobuf - and one bad length must not take down
//! the load.

#[derive(Debug, Clone, Copy)]
pub enum Val {
    Varint(u64),
    F32(f32),
    F64(f64),
    /// byte range into the buffer the walk started from
    Bytes(usize, usize),
}

impl Val {
    pub fn f32(self) -> f32 {
        match self {
            Val::F32(v) => v,
            Val::Varint(v) => v as f32,
            Val::F64(v) => v as f32,
            _ => 0.0,
        }
    }
    pub fn range(self) -> Option<(usize, usize)> {
        match self {
            Val::Bytes(a, b) => Some((a, b)),
            _ => None,
        }
    }
}

fn varint(d: &[u8], i: &mut usize) -> Option<u64> {
    let mut r = 0u64;
    let mut s = 0u32;
    loop {
        let b = *d.get(*i)?;
        *i += 1;
        r |= ((b & 0x7F) as u64) << s;
        if b & 0x80 == 0 {
            return Some(r);
        }
        s += 7;
        if s > 63 {
            return None;
        }
    }
}

/// Walk the fields in `d[start..end]`, calling `f(field_number, value)`.
pub fn fields<F: FnMut(u32, Val)>(d: &[u8], start: usize, end: usize, mut f: F) {
    let end = end.min(d.len());
    let mut i = start;
    while i < end {
        let key = match varint(d, &mut i) {
            Some(k) => k,
            None => return,
        };
        let (fnum, wt) = ((key >> 3) as u32, (key & 7) as u8);
        match wt {
            0 => match varint(d, &mut i) {
                Some(v) => f(fnum, Val::Varint(v)),
                None => return,
            },
            1 => {
                if i + 8 > end {
                    return;
                }
                let v = f64::from_le_bytes(d[i..i + 8].try_into().unwrap());
                i += 8;
                f(fnum, Val::F64(v));
            }
            2 => {
                let n = match varint(d, &mut i) {
                    Some(n) => n as usize,
                    None => return,
                };
                // ⚠ `n` is a full u64 off the wire (a peer's, online): `i + n`
                // wrapped, passed the check, handed out a reversed range and
                // walked `i` backwards - a hang or a slice panic. (`end - i`
                // is no good either: a varint may have read past a
                // sub-message's `end`.)
                let e = match i.checked_add(n) {
                    Some(e) if e <= end => e,
                    _ => return,
                };
                f(fnum, Val::Bytes(i, e));
                i = e;
            }
            5 => {
                if i + 4 > end {
                    return;
                }
                let v = f32::from_le_bytes(d[i..i + 4].try_into().unwrap());
                i += 4;
                f(fnum, Val::F32(v));
            }
            _ => return,
        }
    }
}

pub fn text(d: &[u8], r: (usize, usize)) -> String {
    // a range that is not one of this buffer's is empty, not a panic
    String::from_utf8_lossy(d.get(r.0..r.1).unwrap_or(&[])).into_owned()
}

/// Packed varints, as used for mesh index buffers.
pub fn varints(d: &[u8], start: usize, end: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = start;
    while i < end.min(d.len()) {
        match varint(d, &mut i) {
            Some(v) => out.push(v as u32),
            None => break,
        }
    }
    out
}

/// A 4x4 row-major matrix written as floats 1..16 with proto3 defaults
/// omitted, which is how both bone rest poses and animation keyframes are
/// stored. Missing fields are zero, so the identity has to be supplied by the
/// caller if that is what an empty message should mean.
pub fn matrix(d: &[u8], start: usize, end: usize) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    fields(d, start, end, |fnum, v| {
        if (1..=16).contains(&fnum) {
            m[(fnum - 1) as usize] = v.f32();
        }
    });
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A length that wraps `i + n` (2^64 - 12 after a 2-byte key + 10-byte
    /// varint) ends the walk: no callback, no hang, no reversed range.
    #[test]
    fn huge_length_ends_the_walk() {
        // field 99, wire type 2, length 0xFFFF_FFFF_FFFF_FFF4
        let mut d = vec![0x9A, 0x06];
        d.extend_from_slice(&[0xF4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
        d.extend_from_slice(&[0; 16]);
        let mut seen = 0;
        fields(&d, 0, d.len(), |_, v| {
            seen += 1;
            if let Val::Bytes(a, b) = v {
                assert!(a <= b && b <= d.len());
            }
        });
        assert_eq!(seen, 0);
        // every length from just past the buffer up to the wrap
        for n in [17u64, 1 << 32, u64::MAX - 12, u64::MAX - 11, u64::MAX] {
            let mut d = vec![0x0A];
            let mut v = n;
            while v >= 0x80 {
                d.push(v as u8 | 0x80);
                v >>= 7;
            }
            d.push(v as u8);
            d.extend_from_slice(&[0; 16]);
            fields(&d, 0, d.len(), |_, _| panic!("length {n} accepted"));
        }
    }

    #[test]
    fn truncated_fixed_fields_end_the_walk() {
        fields(&[0x0D, 1, 2, 3], 0, 4, |_, _| panic!("short f32"));
        fields(&[0x09, 1, 2, 3, 4, 5, 6, 7], 0, 8, |_, _| panic!("short f64"));
        let mut n = 0;
        fields(&[0x0D, 0, 0, 0x80, 0x3F, 0x0A, 2, b'h', b'i'], 0, 9, |f, v| {
            n += 1;
            match v {
                Val::F32(x) => assert_eq!((f, x), (1, 1.0)),
                Val::Bytes(a, b) => assert_eq!((f, a, b), (1, 7, 9)),
                _ => panic!(),
            }
        });
        assert_eq!(n, 2);
    }

    /// A sub-message walk whose length varint runs past `end` must not hand
    /// out bytes beyond `end`.
    #[test]
    fn varint_past_end_ends_the_walk() {
        let d = [0x0A, 0x81, 0x00, b'x', b'y'];
        fields(&d, 0, 2, |_, _| panic!("read past end"));
    }

    #[test]
    fn text_of_a_bad_range_is_empty() {
        assert_eq!(text(b"abc", (2, 1)), "");
        assert_eq!(text(b"abc", (1, 9)), "");
        assert_eq!(text(b"abc", (1, 3)), "bc");
    }
}
