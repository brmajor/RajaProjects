/*
  Filename   : Timer.hpp
  Author     : Gary M. Zoppetti
  Course     : Varies
  Assignment : -
  Description: A templated timer class for timing algorithms.
               { steady, system, high_resolution }_clock may be used.
*/

/************************************************************/
// Macro guard to prevent multiple inclusions

#ifndef TIMER_H
#define TIMER_H

/************************************************************/
// System includes

#include <chrono>

/************************************************************/
// Local includes

/************************************************************/
// Using declarations

/************************************************************/

template <typename Clock = std::chrono::steady_clock>
class Timer
{
public:

  Timer ()
  {
    start ();
  }

  void
  start ()
  {
    m_start = Clock::now ();
  }

  void
  stop ()
  {
    m_stop = Clock::now ();
  }

  double
  getElapsedMs () const
  {
    auto timeDelta = m_stop - m_start;
    double elapsedMs = std::chrono::duration
      <double, std::milli> (timeDelta).count ();

    return elapsedMs;
  }

private:

  decltype (Clock::now ()) m_start;
  decltype (Clock::now ()) m_stop;
};

/************************************************************/

#endif

/************************************************************/
