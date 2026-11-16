// Tap-rhythm recorder (CS109 final project)
//
// Wiring: a push button between pin 2 and GND (no resistor needed,
// the internal pull-up is used). The onboard LED lights while pressed.
//
// Each attempt is a burst of taps. When no tap arrives for TIMEOUT_MS,
// the attempt ends and one line is sent over serial:
//     TAPS,<gap1_ms>,<gap2_ms>,...
// where each gap is the time between consecutive presses, in milliseconds.

const int BUTTON_PIN = 2;
const int LED_PIN = LED_BUILTIN;
const unsigned long DEBOUNCE_US = 25000;  // ignore bounces shorter than 25 ms
const unsigned long TIMEOUT_MS = 1500;    // silence that ends an attempt
const int MAX_TAPS = 32;

unsigned long tapTimes[MAX_TAPS];
int nTaps = 0;
int lastState = HIGH;
unsigned long lastChangeUs = 0;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  int state = digitalRead(BUTTON_PIN);
  unsigned long nowUs = micros();

  // Debounced edge detection
  if (state != lastState && (nowUs - lastChangeUs) > DEBOUNCE_US) {
    lastChangeUs = nowUs;
    lastState = state;
    if (state == LOW) {                 // button pressed
      if (nTaps < MAX_TAPS) tapTimes[nTaps++] = nowUs;
      digitalWrite(LED_PIN, HIGH);
    } else {
      digitalWrite(LED_PIN, LOW);
    }
  }

  // End of attempt: report the gaps between taps
  if (nTaps > 0 && (micros() - tapTimes[nTaps - 1]) / 1000UL > TIMEOUT_MS) {
    Serial.print("TAPS");
    for (int i = 1; i < nTaps; i++) {
      Serial.print(',');
      Serial.print((tapTimes[i] - tapTimes[i - 1]) / 1000.0, 1);
    }
    Serial.println();
    nTaps = 0;
  }
}
