#pragma once

#include <predictive-unit/util/tuple.h>
#include <predictive-unit/base.h>
#include <predictive-unit/log.h>

namespace NPredUnit {


	template <typename T>
	class TArgument {
	public:
		TArgument(TString fullName, TString shortName, T& dstValue, TString description, bool required = false, bool stopProcessingAfterMatch = false)
			: FullName(fullName)
			, ShortName(shortName)
			, DstValue(dstValue)
			, Description(description)
			, Required(required)
			, StopProcessingAfterMatch(stopProcessingAfterMatch)
		{
		}

		void GenerateHelp(TOutputStream& ostream) const {
			ostream << "\t" << FullName << ", " << ShortName;
			if (!Description.empty()) {
			 	ostream << ", " << Description;
			}
			ostream << "\n";
		}


		bool Match(const TString& name) {
			return ((name == FullName) || (name == ShortName));
		}

		bool ShouldStopProcessingAfterMatch() const {
			return StopProcessingAfterMatch;
		}
		

		TVector<TString>::const_iterator SetValue(const TVector<TString>& args, TVector<TString>::const_iterator it);
		
	private:
		TString FullName;
		TString ShortName;
		T& DstValue;
		TString Description;
		bool Required;
		bool StopProcessingAfterMatch;
	};

	template <typename T>
	TArgument<T> Argument(
		TString fullName, 
		TString shortName, 
		T& dstValue, 
		TString description = TString(), 
		bool required = false, 
		bool shouldStopProcessingAfterMatch = false
	) {
		return TArgument<T>(fullName, shortName, dstValue, description, required, shouldStopProcessingAfterMatch);
	}


	template <class ... TArgs>
	class TArgumentSet {
	public:
		TArgumentSet(TArgs ... args)
			: Args(args ...)
		{
		}

		bool TryParse(const TVector<TString>& args) {
			try {
				Parse(args);	
			} catch(const TErrException& e) {
				L_ERROR << e.what();
				return false;
			}
			return true;
		}

		void Parse(const TVector<TString>& args) {
			bool stopOptionsProcessing = false;
			auto it = args.begin();
			while (it != args.end() && !stopOptionsProcessing) {
				bool matchFound = false;

				ForEach(Args, [&](auto& opt) {
					if (opt.Match(*it)) {
						if (opt.ShouldStopProcessingAfterMatch()) {
							stopOptionsProcessing = true;
						}
						it = opt.SetValue(args, it);
						matchFound = true;
					}
				});
				ENSURE(matchFound, "Failed to find command line option for: " << *it);
			}
		}

		void GenerateHelp(TOutputStream& ostream) {
			ForEach(Args, [&](auto& opt) {
				opt.GenerateHelp(ostream);
			});
		}

	private:
		std::tuple<TArgs...> Args;
	};

	template <class ... TArgs>
	TArgumentSet<TArgs ...> ArgumentSet(TArgs ... args) {
		return TArgumentSet<TArgs ...>(args ...);
	}

} // namespace NPredUnit
