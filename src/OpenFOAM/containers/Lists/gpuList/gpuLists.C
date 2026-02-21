#include "gpuList.C"

namespace Foam
{
    template class gpuList<bool>;
    template class gpuList<char>;
    template class gpuList<label>;
    template class gpuList<float>;
    template class gpuList<double>;

    // Explicit instantiation of non-member IO operators (needed for SP builds)
    template Ostream& operator<<(Ostream&, const gpuList<label>&);
    template Ostream& operator<<(Ostream&, const gpuList<float>&);
    template Ostream& operator<<(Ostream&, const gpuList<double>&);
    template Istream& operator>>(Istream&, gpuList<label>&);
    template Istream& operator>>(Istream&, gpuList<float>&);
    template Istream& operator>>(Istream&, gpuList<double>&);
}
