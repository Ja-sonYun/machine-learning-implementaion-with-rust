use crate::maths::n_matrix::Matrix;
use std::io::{Error, ErrorKind};
use std::io::prelude::*;
use std::fs::File;
use std::ptr;

pub fn create_matrix_from_image(path: &'static str) -> Matrix<u8> {
    let decoder = png::Decoder::new(File::open(path).unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    // Allocate the output buffer.
    let mut buf = vec![0; info.buffer_size()];
    reader.next_frame(&mut buf).unwrap();

    Matrix::<u8>::from(vec![info.height as usize, info.width as usize], buf)
}

///----------------------Test from here----------------------
/// TODO: understanding IDAT decode method

static PNG_SIGNATURE: &'static [u8] = &[137, 80, 78, 71, 13, 10, 26, 10];
static IHDR_LENGTH:   &'static [u8] = &[0, 0, 0, 13];
static IHDR:          &'static [u8] = &[73, 72, 68, 82];


// struct Chunk {
//     Length: [u8;4], // 4bytes
//     ChunkType: [u8;4],
//     ChunkData: Vec<u8>,
//     CRC: [u8;4],
// }


// MNIST  ->
        // Width: 28
        // Height: 28
        // Bit_depth: [8]
        // Color_Type: [0]
        // Compression_method: [0]
        // Filter_method: [0]
        // Interlace_method: [0]
        // CRC: [87, 102, 128, 72]
        // IDAT[0, 0, 0, 216, 73, 68, 65, 84]
        // zlib comp[120, 156]
        // deflat comp algm [99, 96, 24, 96, 96]

union __TypeCastingFrom4LenTo16 {
    array: [u8;4],
    num: u16,
}
fn _type_cast_u4a_to_u16(a: [u8;4]) -> u16 {
    let mut tmp = a;
    tmp.reverse();
    let t = __TypeCastingFrom4LenTo16 { array: tmp };
    unsafe { t.num }
}

pub fn parse_image(path: &'static str) -> Result<(), Error> {
    let mut f = File::open(path)?;
    let mut Buffer: [u8;8] = [0;8];
    f.read(&mut Buffer)?;

    if Buffer == PNG_SIGNATURE {
        println!(" -- {:?}", Buffer);
    } else {
        return Err(Error::new(ErrorKind::Other, "not png!"))
    }

    // Read ChunkLength and ChunkType
    f.read(&mut Buffer)?;
    let mut ChunkLength: [u8;4] = [0;4];
    let mut ChunkType: [u8;4] = [0;4];

    unsafe {
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr(), ChunkLength.as_mut_ptr(), 4);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(4), ChunkType.as_mut_ptr(), 4);
    }

    if ChunkLength == IHDR_LENGTH && ChunkType == IHDR {
        println!("[*] Found IHDR Chunk!");
    } else {
        return Err(Error::new(ErrorKind::Other, "does not support this png!"))
    }

    // Buffer for read ChunkData and CRC
    let mut Buffer: [u8;17] = [0;17];

    let mut CRC: [u8;4] = [0;4];
    let mut Width: [u8; 4] = [0;4];
    let mut Height: [u8; 4] = [0;4];
    let mut Bit_depth: [u8; 1] = [0;1];
    let mut Color_Type: [u8; 1] = [0;1];
    let mut Compression_method: [u8; 1] = [0;1];
    let mut Filter_method: [u8; 1] = [0;1];
    let mut Interlace_method: [u8; 1] = [0;1];

    f.read(&mut Buffer)?;
    unsafe {
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr(), Width.as_mut_ptr(), 4);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(4), Height.as_mut_ptr(), 4);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(8), Bit_depth.as_mut_ptr(), 1);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(9), Color_Type.as_mut_ptr(), 1);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(10), Compression_method.as_mut_ptr(), 1);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(11), Filter_method.as_mut_ptr(), 1);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(12), Interlace_method.as_mut_ptr(), 1);
        ptr::copy_nonoverlapping::<u8>(Buffer.as_ptr().offset(13), CRC.as_mut_ptr(), 4);
    }

    let Width = _type_cast_u4a_to_u16(Width);
    let Height = _type_cast_u4a_to_u16(Height);

    println!("Width: {:?}\nHeight: {:?}\nBit_depth: {:?}\nColor_Type: {:?}\nCompression_method: {:?}\nFilter_method: {:?}\nInterlace_method: {:?}\nCRC: {:?}", Width, Height, Bit_depth, Color_Type, Compression_method, Filter_method, Interlace_method, CRC);

    if Color_Type != [0] {
        panic!("Only tested with Grayscale images")
    }

    let mut Buffer: [u8;8] = [0;8];
    f.read(&mut Buffer)?;
    println!("IDAT{:?}", Buffer);

    let mut Buffer: [u8;2] = [0;2]; // MNIST has Default Comporession, [78, 9C]
    f.read(&mut Buffer)?;
    println!("zlib comp{:?}", Buffer);

    let mut Buffer: [u8;5] = [0;5];
    f.read(&mut Buffer)?;
    println!("deflat comp algm {:?}", Buffer);

    // MNIST doesn't need Filter ( Filt(x) = Orig(x),  Recom(x) = Filt(x) )
    // Compression method is 0 so that it use LZ77 algorithm
    Ok(())

        //TODO: implement LZ77 algorithm
}
