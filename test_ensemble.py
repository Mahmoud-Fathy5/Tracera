"""Test ensemble detection on fat7y.png and smallMoMedhat.jpg"""
from inference import GramNetDetector

detector = GramNetDetector('model', device='cpu')

print("\n=== fat7y.png (known FAKE) ===")
f = open('fat7y.png', 'rb')
r = detector.predict(f.read())
f.close()
print(r)

print("\n=== smallMoMedhat.jpg (real photo) ===")
f = open('smallMoMedhat.jpg', 'rb')
r = detector.predict(f.read())
f.close()
print(r)
