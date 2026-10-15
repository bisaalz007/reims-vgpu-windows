// Run by osascript in the logged-in guest; no GUI clicks or downloaded tools.
ObjC.import('AppKit');
ObjC.bindFunction('CGDisplayCopyAllDisplayModes', ['void *', ['unsigned int', 'void *']]);
ObjC.bindFunction('CGDisplayCopyDisplayMode', ['void *', ['unsigned int']]);
ObjC.bindFunction('CFArrayGetCount', ['long', ['void *']]);
ObjC.bindFunction('CFArrayGetValueAtIndex', ['void *', ['void *', 'long']]);
ObjC.bindFunction('CGDisplayModeGetPixelWidth', ['unsigned long', ['void *']]);
ObjC.bindFunction('CGDisplayModeGetPixelHeight', ['unsigned long', ['void *']]);
ObjC.bindFunction('CGBeginDisplayConfiguration', ['int', ['void **']]);
ObjC.bindFunction('CGConfigureDisplayWithDisplayMode', ['int', ['void *', 'unsigned int', 'void *', 'void *']]);
ObjC.bindFunction('CGCompleteDisplayConfiguration', ['int', ['void *', 'unsigned int']]);
ObjC.bindFunction('CGCancelDisplayConfiguration', ['int', ['void *']]);
ObjC.bindFunction('CFRelease', ['void', ['void *']]);

function run(args) {
    if (args.length !== 1 || !/^(1280x720|1920x1080|2560x1440|3840x2160)$/.test(args[0])) throw Error('Choose a supported display mode');
    var wanted = args[0].split('x').map(Number);
    var display = $.CGMainDisplayID();
    var modes = $.CGDisplayCopyAllDisplayModes(display, null);
    var selected = null;
    for (var n = 0; n < $.CFArrayGetCount(modes); n++) {
        var mode = $.CFArrayGetValueAtIndex(modes, n);
        if (Number($.CGDisplayModeGetPixelWidth(mode)) === wanted[0] && Number($.CGDisplayModeGetPixelHeight(mode)) === wanted[1]) { selected = mode; break; }
    }
    if (selected === null) throw Error('The guest does not offer this mode');
    var configuration = Ref();
    var result = Number($.CGBeginDisplayConfiguration(configuration));
    if (result !== 0) { $.CFRelease(modes); throw Error('Cannot begin display configuration: ' + result); }
    result = Number($.CGConfigureDisplayWithDisplayMode(configuration[0], display, selected, null));
    $.CFRelease(modes);
    if (result !== 0) { $.CGCancelDisplayConfiguration(configuration[0]); throw Error('Cannot configure display: ' + result); }
    // A direct CGDisplaySetDisplayMode change ends when osascript exits.
    result = Number($.CGCompleteDisplayConfiguration(configuration[0], $.kCGConfigurePermanently));
    if (result !== 0) throw Error('macOS refused the display mode: ' + result);
    var current = $.CGDisplayCopyDisplayMode(display);
    var actual = [Number($.CGDisplayModeGetPixelWidth(current)), Number($.CGDisplayModeGetPixelHeight(current))];
    $.CFRelease(current);
    if (actual[0] !== wanted[0] || actual[1] !== wanted[1]) throw Error('macOS has not applied the display mode');
    return JSON.stringify({display_applied: args[0]});
}
