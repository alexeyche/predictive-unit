#include "argument.h"

#include <predictive-unit/util/optional.h>

namespace NPredUnit {

	template <>
	TVector<TString>::const_iterator TArgument<bool>::SetValue(const TVector<TString>& args, TVector<TString>::const_iterator it) {
		DstValue = true;
		return ++it;
	}

	template <>
	TVector<TString>::const_iterator TArgument<ui32>::SetValue(const TVector<TString>& args, TVector<TString>::const_iterator it) {
		++it;
		ENSURE(it != args.end(), "Need value for command line option: " << FullName);
		DstValue = std::stoi(*it);
		return ++it;
	}

	template <>
	TVector<TString>::const_iterator TArgument<TOptional<int>>::SetValue(const TVector<TString>& args, TVector<TString>::const_iterator it) {
		++it;
		ENSURE(it != args.end(), "Need value for command line option: " << FullName);
		DstValue = std::stoi(*it);
		return ++it;
	}


	template <>
	TVector<TString>::const_iterator TArgument<TString>::SetValue(const TVector<TString>& args, TVector<TString>::const_iterator it) {
		++it;
		ENSURE(it != args.end(), "Need value for command line option: " << FullName);
		DstValue = *it;
		return ++it;
	}
}