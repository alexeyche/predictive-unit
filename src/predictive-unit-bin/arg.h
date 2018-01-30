#pragma once

#include <tuple>
#include <utility>

namespace NPredUnit {

	template<std::size_t I = 0, typename FuncT, typename... Tp>
	inline typename std::enable_if<I == sizeof...(Tp), void>::type
	  ForEach(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
	  { }

	template<std::size_t I = 0, typename FuncT, typename... Tp>
	inline typename std::enable_if<I < sizeof...(Tp), void>::type
	  ForEach(std::tuple<Tp...>& t, FuncT f)
	  {
	    f(std::get<I>(t));
	    ForEach<I + 1, FuncT, Tp...>(t, f);
	  }

	template<std::size_t I = 0, typename FuncT, typename... Tp>
	inline typename std::enable_if<I == sizeof...(Tp), void>::type
	  ForEachEnumerate(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
	  { }

	template<std::size_t I = 0, typename FuncT, typename... Tp>
	inline typename std::enable_if<I < sizeof...(Tp), void>::type
	  ForEachEnumerate(std::tuple<Tp...>& t, FuncT f)
	  {
	    f(I, std::get<I>(t));
	    ForEachEnumerate<I + 1, FuncT, Tp...>(t, f);
	  }

	template <typename T>
	class TArgument {
	public:
		TArgument(const TString& fullName, const TString& shortName, T& dstValue, bool required = false)
			: FullName(fullName)
			, ShortName(shortName)
			, DstValue(dstValue)
			, Required(required)
		{
		}

		const TString& GetFullName() const {
			return FullName;
		}


		bool Match(const string& name) {
			return ((name == FullName) || (name == ShortName));
		}

	private:
		TString FullName;
		TString ShortName;
		T& DstValue;
		bool Required;
	};

	template <class ... TArgs>
	class TArgumentSet {
	public:
		TArgumentSet(TArgs ... args)
			: Args(args ...)
		{
		}

		void Parse(int argc, const char** argv) {
			TVector<TString> args;
			int argIt = 0;
			while (argIt < argc) {
				args.push_back(argv[argIt]);
				++argIt;
			}
			for (auto it=args.begin(); it != args.end(); ++it) {
				ForEach(Args, [&](auto& opt) {
					if (opt.Match(*it)) {
						L_INFO << "Opt " << opt.GetFullName() << " matched to " << *it;
					}
				});
			}
		}

	private:
		std::tuple<TArgs...> Args;
	};

	template <class ... TArgs>
	TArgumentSet<TArgs ...> ArgumentSet(TArgs ... args) {
		return TArgumentSet<TArgs ...>(args ...);
	}

	template <typename T>
	TArgument<T> Argument(const TString& fullName, const TString& shortName, T& dstValue, bool required = false) {
		return TArgument<T>(fullName, shortName, dstValue, required);
	}

} // namespace NPredUnit
