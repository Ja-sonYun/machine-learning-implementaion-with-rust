use time::OffsetDateTime;

pub fn test_excution_time<F: Fn()>(f: F) {
    let now2 = OffsetDateTime::now_utc();
    f();
    println!("excution time: {:?}", OffsetDateTime::now_utc() - now2);
}
