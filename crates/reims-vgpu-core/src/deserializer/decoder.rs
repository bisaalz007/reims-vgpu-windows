//! The decoders, one per encoder type: render (0), compute (1), blit (2) and info (4).
//!
//! The interface the walk drives: [`Decoder::for_segment`] makes the decoder of a segment's
//! type, [`Decoder::decode`] runs one segment through it, and [`Decoder::fault`] ends its
//! encoder after a raise. A decoder that raises out of `decode` has already faulted itself;
//! `fault` may raise too (a Metal `endEncoding` raise), and its raise goes on in place of the
//! one in flight.

use std::sync::Weak;

use crate::deserializer::blit::BlitDecoder;
use crate::deserializer::compute::ComputeDecoder;
use crate::deserializer::info::InfoDecoder;
use crate::deserializer::records::SegmentHeader;
use crate::deserializer::render::RenderDecoder;
use crate::deserializer::{Deserializer, RaiseKind};
use crate::memory::cursor::Cursor;
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::rt::Hold;

/// A segment's decoder. It lives inside the caller's unit of work (its real pool) and may be
/// carried into the next segment (`continued_by_next`).
pub enum Decoder<P: Platform> {
    /// Encoder type 0.
    Render(RenderDecoder<P>),
    /// Encoder type 1.
    Compute(ComputeDecoder<P>),
    /// Encoder type 2.
    Blit(BlitDecoder<P>),
    /// Encoder type 4.
    Info(InfoDecoder<P>),
    /// A scripted decoder for the walk's unit tests (`mock`).
    #[cfg(test)]
    Mock {
        /// The scripted decoder.
        decoder: mock::MockDecoder,
    },
}

impl<P: Platform> Decoder<P> {
    /// The decoder of `encoder_type`, holding `deserializer` weakly.
    ///
    /// `command_buffer` goes to the render, compute and blit decoders as it is, `None`
    /// included: render and compute keep it and later fail to make their encoder, and blit asks
    /// it for its encoder at once. Render and compute hold their own reference to it, which the
    /// decoder's release (when the caller lets the walk's decoders go) drops before the caller's
    /// own older reference to the buffer, so never the last.
    pub(crate) fn for_segment(
        deserializer: Weak<Deserializer<P>>,
        encoder_type: u8,
        command_buffer: Option<&P::CommandBuffer>,
    ) -> RaiseResult<Decoder<P>> {
        #[cfg(test)]
        if let Some(m) = mock::make(encoder_type) {
            return Ok(Decoder::Mock { decoder: m });
        }
        Ok(match encoder_type {
            0 => Decoder::Render(RenderDecoder::new(
                deserializer,
                command_buffer.map(Handle::clone_ref),
            )),
            1 => Decoder::Compute(ComputeDecoder::new(
                deserializer,
                command_buffer.map(Handle::clone_ref),
            )),
            2 => Decoder::Blit(BlitDecoder::new(deserializer, command_buffer)?),
            4 => Decoder::Info(InfoDecoder::new(deserializer)),
            other => crate::raise!(
                RaiseKind::EncoderType,
                format_args!("unknown encoder type {other}")
            ),
        })
    }

    /// The encoder type it decodes.
    pub fn encoder_type(&self) -> u8 {
        match self {
            Decoder::Render(_) => 0,
            Decoder::Compute(_) => 1,
            Decoder::Blit(_) => 2,
            Decoder::Info(_) => 4,
            #[cfg(test)]
            Decoder::Mock { decoder } => decoder.type_id,
        }
    }

    /// Decodes one segment. A raise out of it has faulted the decoder already.
    pub fn decode(
        &mut self,
        hold: Hold<'_>,
        header: &SegmentHeader,
        segment: Cursor,
    ) -> RaiseResult<()> {
        match self {
            Decoder::Render(d) => d.decode(hold, header, segment),
            Decoder::Compute(d) => d.decode(hold, header, segment),
            Decoder::Blit(d) => d.decode(hold, header, segment),
            Decoder::Info(d) => d.decode(hold, header, segment),
            #[cfg(test)]
            Decoder::Mock { decoder } => decoder.decode(header, segment),
        }
    }

    /// Ends the decoder's encoder after a raise in a later segment. A raise from the encoder's
    /// `endEncoding` stops the fault there (the encoder, and render's pipeline, stay set) and
    /// goes on in place of the raise in flight.
    pub fn fault(&mut self) -> RaiseResult<()> {
        match self {
            Decoder::Render(d) => d.fault(),
            Decoder::Compute(d) => d.fault(),
            Decoder::Blit(d) => d.fault(),
            // No encoder: nothing to end.
            Decoder::Info(_) => Ok(()),
            #[cfg(test)]
            Decoder::Mock { decoder } => decoder.fault(),
        }
    }
}

/// A scripted decoder standing in for the four in the walk's unit tests: it records what it is
/// asked, and a segment whose first payload word is [`mock::RAISE`] makes it raise as a decoder
/// does (its own fault, then the raise).
#[cfg(test)]
pub(crate) mod mock {

    use std::cell::RefCell;
    use std::sync::Arc;

    use crate::deserializer::RaiseKind;
    use crate::deserializer::records::SegmentHeader;
    use crate::memory::cursor::Cursor;
    use crate::raise::{Raise, RaiseResult};
    use crate::sync::Mutex;

    /// A payload word that makes the mock raise.
    pub const RAISE: u32 = 0xdead_0001;

    /// A payload word that makes the mock's later fault raise (as an `endEncoding` raising
    /// inside a decoder's fault does).
    pub const FAULT_RAISES: u32 = 0xdead_0002;

    /// A payload word that makes the mock raise as [`RAISE`] does, its raise carrying a value
    /// (as a landing frame's raise carries what the frame held).
    pub const RAISE_CARRYING: u32 = 0xdead_0003;

    /// What the mocks did, in order: `new <id> <type>`, `decode <id> <type> <cont_prev>
    /// <cont_next> <remaining> <word>`, `fault <id>`, `drop <id>`, and `release <id>` when the
    /// value a mock's raise carried is released.
    pub type Events = Arc<Mutex<Vec<String>>>;

    /// The value a [`RAISE_CARRYING`] raise carries: notes its release.
    struct Carried {
        id: u32,
        events: Events,
    }

    impl Drop for Carried {
        fn drop(&mut self) {
            self.events.lock().push(format!("release {}", self.id));
        }
    }

    /// One mock decoder.
    pub struct MockDecoder {
        /// Its identity in `events`.
        pub id: u32,
        /// Its encoder type.
        pub type_id: u8,
        /// Whether `fault` raises (set by a [`FAULT_RAISES`] payload word).
        pub fault_raises: bool,
        events: Events,
    }

    impl MockDecoder {
        pub(super) fn decode(
            &mut self,
            header: &SegmentHeader,
            mut segment: Cursor,
        ) -> RaiseResult<()> {
            let remaining = segment.remaining();
            let word = segment.read::<u32>().ok();
            self.events.lock().push(format!(
                "decode {} {} {} {} {} {:?}",
                self.id,
                header.encoder_type,
                header.continues_previous,
                header.continued_by_next,
                remaining,
                word
            ));
            if word == Some(FAULT_RAISES) {
                self.fault_raises = true;
            }
            if word == Some(RAISE) || word == Some(RAISE_CARRYING) {
                let mut raised = Raise::new(RaiseKind::Mock("mock.decode"));
                if word == Some(RAISE_CARRYING) {
                    raised = raised.defer(Carried {
                        id: self.id,
                        events: self.events.clone(),
                    });
                }
                // A decoder's own handler: the raise, its fault, then on.
                let r: RaiseResult<()> = Err(raised);
                self.events.lock().push(format!("fault {}", self.id));
                return r;
            }
            Ok(())
        }

        pub(super) fn fault(&mut self) -> RaiseResult<()> {
            self.events.lock().push(format!("fault {}", self.id));
            if self.fault_raises {
                return Err(Raise::new(RaiseKind::Mock("mock.fault")));
            }
            Ok(())
        }
    }

    impl Drop for MockDecoder {
        fn drop(&mut self) {
            self.events.lock().push(format!("drop {}", self.id));
        }
    }

    type Factory = Box<dyn FnMut(u8) -> Option<MockDecoder>>;

    thread_local! {
        static FACTORY: RefCell<Option<Factory>> = const { RefCell::new(None) };
    }

    /// The decoder `for_segment` makes for `encoder_type` while a factory is installed.
    pub(super) fn make(encoder_type: u8) -> Option<MockDecoder> {
        FACTORY.with(|f| f.borrow_mut().as_mut().and_then(|f| f(encoder_type)))
    }

    /// Installs a factory on this thread making a mock for the types 0, 1, 2 and 4 (the others
    /// fall through to the real match, which refuses them), numbered from 1; returns the shared
    /// event log. `uninstall` removes it.
    pub fn install() -> Events {
        let events: Events = Arc::new(Mutex::new(Vec::new()));
        let log = events.clone();
        let mut next = 0u32;
        let factory: Factory = Box::new(move |t| {
            if !matches!(t, 0 | 1 | 2 | 4) {
                return None;
            }
            next = next.wrapping_add(1);
            log.lock().push(format!("new {next} {t}"));
            Some(MockDecoder {
                id: next,
                type_id: t,
                fault_raises: false,
                events: log.clone(),
            })
        });
        FACTORY.with(|f| *f.borrow_mut() = Some(factory));
        events
    }

    /// Removes this thread's factory.
    pub fn uninstall() {
        FACTORY.with(|f| *f.borrow_mut() = None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::Hold;
    use std::sync::Arc;

    use crate::deserializer::test_lookups::Lookups;
    use crate::raise::RaiseId;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::Model;
    use crate::testkit::{FakePlatform, Fk, kind};

    type P = FakePlatform;

    fn setup() -> (
        Arc<FakePlatform>,
        Arc<Deserializer<P>>,
        Fk<kind::CommandBuffer>,
    ) {
        let p = Arc::new(FakePlatform::new());
        let r = Arc::new(Lookups::new(&p));
        let dev = p.device();
        let d = Deserializer::new(p.clone(), Some(dev), r.lookup());
        let cb: Fk<kind::CommandBuffer> = p.world().new_obj(Model::None);
        p.call_log().clear();
        (p, d, cb)
    }

    fn make(
        d: &Arc<Deserializer<P>>,
        t: u8,
        cb: Option<&Fk<kind::CommandBuffer>>,
    ) -> RaiseResult<Decoder<P>> {
        Decoder::for_segment(Arc::downgrade(d), t, cb)
    }

    #[test]
    fn each_type_makes_its_decoder() {
        let (p, d, cb) = setup();
        for t in [0u8, 1, 2, 4] {
            let dec = make(&d, t, Some(&cb)).unwrap();
            assert_eq!(dec.encoder_type(), t);
            let kind_ok = matches!(
                (&dec, t),
                (Decoder::Render(_), 0)
                    | (Decoder::Compute(_), 1)
                    | (Decoder::Blit(_), 2)
                    | (Decoder::Info(_), 4)
            );
            assert!(kind_ok, "type {t}");
        }
        // Only the blit decoder sends to the command buffer (its encoder, made at once).
        assert_eq!(p.call_log().sels(), vec!["blitCommandEncoder"]);
        assert_eq!(p.call_log().calls()[0].obj, cb.id());
    }

    #[test]
    fn each_decoder_holds_the_deserializer_weakly() {
        // Live decoders leave the caller's reference the only one.
        let (_p, d, cb) = setup();
        let decoders: Vec<Decoder<P>> = [0u8, 1, 2, 4]
            .into_iter()
            .map(|t| make(&d, t, Some(&cb)).unwrap())
            .collect();
        assert_eq!(decoders.len(), 4);
        assert_eq!(Arc::strong_count(&d), 1);
    }

    #[test]
    fn other_types_are_invalid() {
        let (p, d, cb) = setup();
        for t in [3u8, 5, 6, 0x80, 0xff] {
            let e = make(&d, t, Some(&cb)).err().unwrap();
            assert_eq!(e.id(), RaiseId::from(RaiseKind::EncoderType));
        }
        assert!(p.call_log().is_empty());
    }

    #[test]
    fn info_takes_no_command_buffer() {
        let (_p, d, _cb) = setup();
        assert_eq!(make(&d, 4, None).unwrap().encoder_type(), 4);
    }

    #[test]
    fn a_missing_command_buffer_is_passed_on() {
        // Nothing checks it before the decoders take it.
        for t in [0u8, 1, 2] {
            let (_p, d, _cb) = setup();
            let weak = Arc::downgrade(&d);
            let r = catch_death(|| Decoder::for_segment(weak, t, None).map(|d| d.encoder_type()));
            assert_eq!(r.ok().and_then(Result::ok), Some(t));
        }
    }

    #[test]
    fn fault_and_decode_dispatch_to_the_decoder() {
        let (p, d, cb) = setup();
        let mut blit = make(&d, 2, Some(&cb)).unwrap();
        let mark = p.call_log().mark();
        blit.fault().unwrap();
        assert_eq!(p.call_log().sels_since(mark), vec!["endEncoding"]);
        // A second fault has no encoder to end.
        blit.fault().unwrap();
        assert_eq!(p.call_log().sels_since(mark), vec!["endEncoding"]);
        let mut info = make(&d, 4, None).unwrap();
        let mark = p.call_log().mark();
        info.fault().unwrap();
        assert!(p.call_log().sels_since(mark).is_empty());
        let header = SegmentHeader {
            length: 8,
            encoder_type: 4,
            ..SegmentHeader::default()
        };
        info.decode(Hold::NONE, &header, Cursor::default()).unwrap();
    }
}
