#include <Array.hpp>

using namespace std;
using namespace ly;
using namespace cv;

int main(){
    RollingArray<int> ri(10);
    for(int i=0;i<10;i++){
        ri.update(i);
    }
    cout << "test int: " << ri.getMetric() * 1.0 / ri.getSize() << endl;

    RollingArray<float> rf(10);
    for(int i=0;i<10;i++){
        rf.update(i);
    }
    cout << "test float: " << rf.getMetric() * 1.0 / rf.getSize() << endl;

    RollingArray<double> rd(10);
    for(int i=0;i<10;i++){
        rd.update(i);
    }
    cout << "test double: " << rd.getMetric() * 1.0 / rd.getSize() << endl;

    RollingArray<Point3d> rp(10);
    for(int i=0;i<10;i++){
        rp.update(Point3d(i, i, i));
    }
    cout << "test Point3d: " << rp.getMetric() * 1.0 / rp.getSize() << endl;
}