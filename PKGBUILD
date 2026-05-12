# Maintainer: sinoosoida <sinoosoidapass@gmail.com>
pkgname=video-match-finder
pkgver=0.1.0
pkgrel=1
pkgdesc="Find overlapping fragments across a video collection (DINOv2 + FAISS)"
arch=('any')
url="https://github.com/sinoosoida/video-match-finder"
license=('MIT')
depends=(
    'python>=3.10'
    'ffmpeg'
    'python-pytorch'
    'python-torchvision'
    'python-numpy'
    'python-faiss'
    'python-pillow'
    'python-psutil'
    'python-typer'
    'python-rich'
    'python-tqdm'
)
makedepends=('python-build' 'python-installer' 'python-setuptools' 'python-wheel')
source=("$pkgname-$pkgver.tar.gz::https://github.com/sinoosoida/video-match-finder/archive/v$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
    cd "$pkgname-$pkgver"
    python -m build --wheel --no-isolation
}

package() {
    cd "$pkgname-$pkgver"
    python -m installer --destdir="$pkgdir" dist/*.whl
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE" 2>/dev/null || true
}
